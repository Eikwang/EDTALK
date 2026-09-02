"""Quick sync diagnostic — checks Audio2Lip T value and mel alignment."""
import os
import sys
import time
import wave
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from realtime_serve.audio_processor import AudioProcessor, parse_audio_length
from audio import get_hop_size


def quick_diagnostic(wav_path: str):
    print(f"\n{'='*60}")
    print(f"QUICK SYNC DIAGNOSTIC — Mel Spectrogram Alignment Check")
    print(f"{'='*60}")

    # Read WAV
    print(f"\n[1] Reading WAV: {os.path.basename(wav_path)}")
    with wave.open(wav_path, 'rb') as wf:
        framerate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        pcm_bytes = wf.readframes(wf.getnframes())

    print(f"    Original: {framerate}Hz, {channels}ch, {sampwidth*8}bit")

    if sampwidth == 2:
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    else:
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.int16)

    if channels > 1:
        pcm_int16 = pcm_int16.reshape(-1, channels)[:, 0]

    if framerate != 16000:
        from scipy import signal as scipy_signal
        ratio = 16000 / framerate
        n_out = int(len(pcm_int16) * ratio)
        pcm_int16 = scipy_signal.resample(pcm_int16, n_out).astype(np.int16)

    pcm_float = pcm_int16.astype(np.float32) / 32768.0
    pcm_dur = len(pcm_float) / 16000.0
    print(f"    PCM: {len(pcm_float)} samples = {pcm_dur:.4f}s @ 16kHz")

    # Check hop_size and mel_fps
    hop = get_hop_size()
    mel_fps = 16000.0 / hop
    print(f"\n[2] Mel parameters: hop_size={hop}, mel_fps={mel_fps:.2f}")
    print(f"    OLD formula assumed mel_fps=80.0 (hardcoded)")
    print(f"    CORRECT mel_fps={mel_fps:.2f}")

    # Audio2Lip T value
    print(f"\n[3] Audio2Lip timing:")
    wav_length, num_frames = parse_audio_length(len(pcm_float), 16000, 25)
    print(f"    parse_audio_length: num_frames={num_frames}, wav_length={wav_length}")
    print(f"    Lip duration: {num_frames/25:.4f}s")

    processor = AudioProcessor()
    t0 = time.perf_counter()
    mel_input, bs, T = processor.pcm_to_mel(pcm_float)
    t_mel = time.perf_counter() - t0
    print(f"    pcm_to_mel: T={T}, time={t_mel:.2f}s")
    print(f"    T matches num_frames: {T == num_frames}")

    # Check sliding window alignment (OLD formula for comparison)
    print(f"\n[4] Sliding window alignment comparison:")
    syncnet_mel_step_size = 16

    # Compute mel spectrogram directly to count frames
    import audio as audio_module
    cropped_pcm = pcm_float[:wav_length]
    mel = audio_module.melspectrogram(cropped_pcm).T
    mel_total = mel.shape[0]
    print(f"    Mel shape: {mel.shape}, total={mel_total}")
    print(f"    Audio per mel frame: {wav_length/mel_total:.1f} samples = {wav_length*1000/mel_total/16000:.1f}ms")
    print(f"    Mel fps from spectrogram: {mel_total / (wav_length/16000):.2f}")

    print(f"\n    {'Frame':>8} {'OLD start_idx':>14} {'NEW start_idx':>14} {'Diff':>8} {'NEW mel time':>14}")
    for frame_idx in [0, 1, 100, T//2, T-100, T-1]:
        start_frame_num = frame_idx - 2

        # OLD: hardcoded 80
        old_start = int(80.0 * (start_frame_num / 25.0))

        # NEW: actual mel_fps
        new_start = int(mel_fps * (start_frame_num / 25.0))
        new_start = max(0, min(new_start, mel_total - 1))

        diff = new_start - old_start
        mel_time = new_start / mel_fps
        audio_time = frame_idx * 640 / 16000
        offset_ms = (mel_time - audio_time) * 1000

        print(f"    {frame_idx:4d}@{audio_time:.2f}s  {old_start:>8} {new_start:>8} {diff:>+8} "
              f"mel_t={mel_time:.3f}s  offset={offset_ms:.1f}ms")

    # Check if the NEW formula avoids running past the end
    print(f"\n[5] End-of-spectrogram check:")
    last_start = int(mel_fps * ((T-1-2) / 25.0))
    last_end = last_start + syncnet_mel_step_size
    print(f"    Last frame (T={T}): start_idx={last_start}, end_idx={last_end}")
    print(f"    Mel total frames: {mel_total}")
    if last_end > mel_total:
        leftover_ratio = (last_end - mel_total) / syncnet_mel_step_size
        print(f"    OVERFLOW! Last {max(0, T - int((mel_total - 16) / (mel_fps/25)))} frames clamped")
        print(f"    ≈ {max(0, (T - int((mel_total - 16) / (mel_fps/25))) * 40):.0f}ms of audio at end clamped")
    else:
        remaining = mel_total - last_end
        print(f"    OK: {remaining} mel frames remaining after last window")

    # OLD formula comparison
    old_last_start = int(80.0 * ((T-1-2) / 25.0))
    print(f"    OLD formula: start_idx={old_last_start}, end_idx={old_last_start + 16}")
    print(f"    OLD overflow: {old_last_start + 16 - mel_total} frames past end")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  hop_size={hop}, mel_fps={mel_fps:.1f} (was hardcoded as 80.0)")
    print(f"  FIX: start_idx = int({mel_fps:.1f} * (frame_num / 25))")
    print(f"  Old formula caused cumulative drift of {old_last_start - last_start} mel frames")
    print(f"  That's {(old_last_start - last_start) / mel_fps:.1f} seconds of temporal misalignment!")
    print(f"  FIX applied to: realtime_serve/audio_processor.py")


if __name__ == "__main__":
    quick_diagnostic("test_data/wy2.wav")
