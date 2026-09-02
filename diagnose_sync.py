"""Offline audio-lip sync diagnostic test.

Takes the same WAV audio file, runs through the EXACT same pipeline
as the realtime server (pcm_to_mel → Audio2Lip → Generator), and
creates a timestamped video for sync verification.

Usage:
    python diagnose_sync.py <wav_path> [--output output.mp4]

This isolates whether the desync is in:
  A) The Audio2Lip preprocessing/output (T mismatch, mel extraction)
  B) The realtime sync logic (wall-clock timing, frame rate control)

If the offline video is IN SYNC, the issue is in (B).
If the offline video is OUT OF SYNC, the issue is in (A).
"""
import argparse
import os
import time
import wave
import sys

import cv2
import numpy as np
import torch

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from realtime_serve.models import ModelManager
from realtime_serve.audio_processor import AudioProcessor


def run_diagnostic(wav_path: str, character_dir: str, output: str = "diagnose_output.mp4"):
    """Run the diagnostic test.

    Steps:
    1. Read WAV → float32 PCM
    2. pcm_to_mel → mel_input, bs, T
    3. Audio2Lip → lip_coeffs [T, 20]
    4. For each lip coefficient, run Generator inference
    5. Write video with frame timestamp overlay
    """
    # --- 1. Read WAV ---
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: Audio-Lip Sync Offline Test")
    print(f"{'='*60}")
    print(f"\n[1] Reading WAV: {wav_path}")

    with wave.open(wav_path, 'rb') as wf:
        framerate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        pcm_bytes = wf.readframes(wf.getnframes())

    duration_s = len(pcm_bytes) / (framerate * channels * sampwidth)
    print(f"    Sample rate: {framerate}Hz, channels: {channels}, bits: {sampwidth*8}")
    print(f"    Duration: {duration_s:.3f}s, frames: {wf.getnframes()}")

    # Convert to float32
    if sampwidth == 2:
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    else:
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.int16)

    # Handle multi-channel
    if channels > 1:
        pcm_int16 = pcm_int16.reshape(-1, channels)[:, 0]

    # Resample to 16kHz if needed
    if framerate != 16000:
        from scipy import signal as scipy_signal
        ratio = 16000 / framerate
        n_out = int(len(pcm_int16) * ratio)
        pcm_int16 = scipy_signal.resample(pcm_int16, n_out).astype(np.int16)

    pcm_float = pcm_int16.astype(np.float32) / 32768.0
    pcm_dur = len(pcm_float) / 16000.0
    print(f"    PCM float32: {len(pcm_float)} samples, {pcm_dur:.3f}s")

    # --- 2. pcm_to_mel ---
    print(f"\n[2] Converting PCM to mel spectrogram...")
    t0 = time.perf_counter()
    audio_processor = AudioProcessor()
    mel_input, bs, T = audio_processor.pcm_to_mel(pcm_float)
    t_mel = time.perf_counter() - t0
    print(f"    Mel shape: {mel_input.shape}, bs={bs}, T={T}")
    print(f"    Mel computation time: {t_mel:.2f}s")
    print(f"    Lip duration at 25fps: {T/25:.3f}s")
    print(f"    PCM duration: {pcm_dur:.3f}s")
    print(f"    Expected T at 25fps: {int(pcm_dur * 25)}")
    print(f"    Diff: {(T - int(pcm_dur * 25))} frames = {(T - int(pcm_dur * 25)) * 40:.1f}ms")

    # --- 3. Load models ---
    print(f"\n[3] Loading models...")
    model_paths = {
        'audio2lip': os.path.join(character_dir, 'checkpoint', 'audio2lip.pt'),
        'generator': os.path.join(character_dir, 'checkpoint', 'new.pt'),
    }
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"    ERROR: Model not found: {path}")
            return
        print(f"    {name}: {path}")

    t0 = time.perf_counter()
    model_manager = ModelManager(
        model_paths['generator'],
        model_paths['audio2lip'],
        device='cuda'
    )
    model_manager.load()
    t_load = time.perf_counter() - t0
    print(f"    Model loading time: {t_load:.2f}s")

    # --- 4. Audio2Lip inference ---
    print(f"\n[4] Running Audio2Lip inference (T={T})...")
    t0 = time.perf_counter()
    mel_input_flat = mel_input.view(-1, 1, 80, 16)
    lip_coeffs = model_manager.infer_lip(mel_input_flat, bs, T)
    t_infer = time.perf_counter() - t0
    print(f"    Lip coefficients shape: {lip_coeffs.shape}")
    print(f"    Inference time: {t_infer:.2f}s ({T/t_infer:.1f} fps)")
    print(f"    Total lip duration: {lip_coeffs.shape[0] / 25:.3f}s")

    # --- 5. Load character data for Generator input ---
    print(f"\n[5] Loading character data...")
    from realtime_serve.sequence_frame_pack import CharacterDataPack
    character = CharacterDataPack(character_dir)
    current_segment = character.current_segment
    print(f"    Segment: {current_segment}, frames: {character.total_frames}")

    # --- 6. Render all frames with Generator ---
    print(f"\n[6] Rendering all {T} frames with Generator...")
    t0 = time.perf_counter()

    # Pre-compute video frames for quick access
    display_width = 512
    display_height = 512

    # Use av for video writing
    try:
        import av
        from fractions import Fraction
        has_av = True
    except ImportError:
        has_av = False
        print("    WARNING: av not available, using OpenCV VideoWriter")

    if has_av:
        container = av.open(output, mode='w')
        stream = container.add_stream('libx264', rate=Fraction(25, 1))
        stream.width = display_width
        stream.height = display_height
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = 2000000
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output, fourcc, 25, (display_width, display_height))

    frame_times = []
    gen_times = []

    for i in range(T):
        t_gen_start = time.perf_counter()

        # Get character frame (matches demo: vid_idx = i % total_frames)
        orifra, cropped, meta, facemark = character.get_frame(i)

        # Generator inference (same as process_batch in render_loop)
        rendered = model_manager.infer_frame(cropped, lip_coeffs[i:i+1], cropped)
        rendered_np = rendered.clamp(-1, 1).cpu()
        rendered_np = (rendered_np[0] * 127.5 + 127.5).permute(1, 2, 0).to(torch.uint8).numpy()
        rendered_np = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)

        # Resize for display
        rendered_resized = cv2.resize(rendered_np, (display_width, display_height))

        # Overlay frame info
        cv2.putText(rendered_resized, f"Frame: {i}/{T}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(rendered_resized, f"Time: {i/25:.2f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(rendered_resized, f"PCM: {(i*640)/16000:.2f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        gen_time = time.perf_counter() - t_gen_start
        gen_times.append(gen_time)
        frame_times.append(time.perf_counter() - t0)

        if has_av:
            # Convert BGR to RGB for av
            frame_rgb = cv2.cvtColor(rendered_resized, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            av_frame.pts = i  # Frame index as PTS
            av_frame.time_base = Fraction(1, 25)
            for packet in stream.encode(av_frame):
                container.mux(packet)
        else:
            video_writer.write(rendered_resized)

        if i % 125 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            avg_gen = np.mean(gen_times[-125:]) * 1000
            total_elapsed = i / 25
            print(f"    Frame {i}/{T} ({i*100/T:.0f}%), "
                  f"lip_time={total_elapsed:.1f}s, "
                  f"render_time={elapsed:.1f}s, "
                  f"speed={total_elapsed/elapsed:.1f}x realtime, "
                  f"avg_gen={avg_gen:.1f}ms/frame")

    # Flush
    if has_av:
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    else:
        video_writer.release()

    t_total = time.perf_counter() - t0
    avg_gen = np.mean(gen_times) * 1000
    max_gen = np.max(gen_times) * 1000
    print(f"\n    Video written to: {output}")
    print(f"    Total render time: {t_total:.1f}s")
    print(f"    Render speed: {T/25/t_total:.1f}x realtime")
    print(f"    Avg Generator time: {avg_gen:.1f}ms/frame")
    print(f"    Max Generator time: {max_gen:.1f}ms/frame")

    # --- 7. Extract audio for comparison ---
    audio_output = output.replace('.mp4', '.wav')
    print(f"\n[7] Extracting audio...")
    pcm_int16_out = (pcm_float[:T * 640] * 32768).astype(np.int16)
    with wave.open(audio_output, 'wb') as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(16000)
        wf_out.writeframes(pcm_int16_out.tobytes())
    print(f"    Audio written to: {audio_output}")
    print(f"    Audio duration: {len(pcm_int16_out)/16000:.3f}s")

    # --- 8. Summary ---
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC RESULTS")
    print(f"{'='*60}")
    print(f"  Input audio:  {pcm_dur:.3f}s ({len(pcm_float)} samples @ 16kHz)")
    print(f"  Lip frames:   {T} frames = {T/25:.3f}s @ 25fps")
    print(f"  Audio2Lip T:  {T}")
    print(f"  Expected T:   {int(pcm_dur * 25)} ({(int(pcm_dur * 25) * 640)} samples)")
    print(f"  Difference:   {(T - int(pcm_dur * 25))} frames ({(T - int(pcm_dur * 25)) * 40:.1f}ms)")
    print(f"  Audio2Lip time: {t_infer:.2f}s")
    print(f"  Lip coefficients shape: {lip_coeffs.shape}")
    print(f"")
    print(f"  OUTPUT: {output} (video) + {audio_output} (audio)")
    print(f"  Run: ffmpeg -i {output} -i {audio_output} -c:v copy -c:a aac combined.mp4")
    print(f"  Or play separately and check sync manually.")
    print(f"")
    print(f"  VERIFICATION:")
    print(f"  - Video timestamp at frame N: {N/25:.2f}s")
    print(f"  - Audio timestamp at frame N: {N*640/16000:.2f}s")
    print(f"  - These should match within 0.04s (1 frame)")

    return {
        'pcm_dur': pcm_dur,
        'T': T,
        'expected_T': int(pcm_dur * 25),
        'difference_frames': T - int(pcm_dur * 25),
        'difference_ms': (T - int(pcm_dur * 25)) * 40,
        'output_video': output,
        'output_audio': audio_output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose audio-lip sync")
    parser.add_argument("wav_path", help="Path to WAV audio file")
    parser.add_argument("--output", default="diagnose_output.mp4",
                        help="Output video path")
    parser.add_argument("--character-dir", default="test_data/wy",
                        help="Character directory")
    args = parser.parse_args()

    run_diagnostic(args.wav_path, args.character_dir, args.output)
