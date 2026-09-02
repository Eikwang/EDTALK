"""Push a WAV audio file to the realtime inference service.

Usage:
    python push_audio.py <wav_file> [--host localhost] [--port 8000]

Audio format: 16kHz, mono, 16bit PCM. Other formats will be auto-converted.

Push strategy:
    The entire audio file is pushed in a single HTTP request to /audio/push_full.
    The server loads the full PCM into its AudioBuffer, and the render loop
    consumes it in 200ms chunks at its own pace. This ensures:
    - Gapless audio playback (no chunk-boundary artifacts)
    - Continuous lip animation (no idle gaps between chunks)
    - Fast push (one HTTP request instead of many)

    For streaming TTS integration, use /audio/push with chunked mode instead.
"""
import argparse
import base64
import time
import wave

import numpy as np
import requests


def push_wav(wav_path: str, host: str = "localhost", port: int = 8000):
    """Push a WAV file to the /audio/push_full endpoint.

    Sends the entire audio in one request. The server clears its buffer
    and loads the full segment, ensuring gapless playback.

    Args:
        wav_path: Path to WAV audio file
        host: Server host
        port: Server port
    """
    url = f"http://{host}:{port}/audio/push_full"

    # Read WAV
    with wave.open(wav_path, 'rb') as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        pcm_bytes = wf.readframes(nframes)

    duration_s = len(pcm_bytes) / (framerate * channels * sampwidth)
    print(f"WAV info: {framerate}Hz, {channels}ch, {sampwidth*8}bit, {duration_s:.2f}s")

    # Convert to 16kHz mono 16bit if needed
    if framerate != 16000 or channels != 1 or sampwidth != 2:
        print(f"Converting: {framerate}Hz/{channels}ch/{sampwidth*8}bit → 16000Hz/1ch/16bit...")
        pcm_np = np.frombuffer(pcm_bytes, dtype=np.int16 if sampwidth == 2 else np.int32)
        # Reshape for multi-channel
        if channels > 1:
            pcm_np = pcm_np.reshape(-1, channels)[:, 0]  # Take first channel
        # Resample to 16kHz
        if framerate != 16000:
            ratio = 16000 / framerate
            n_out = int(len(pcm_np) * ratio)
            indices = np.linspace(0, len(pcm_np) - 1, n_out).astype(int)
            pcm_np = pcm_np[indices]
        pcm_bytes = pcm_np.astype(np.int16).tobytes()
        duration_s = len(pcm_bytes) / 32000
        print(f"Converted: {duration_s:.2f}s")

    # Push entire audio in one request
    payload = {
        "pcm_base64": base64.b64encode(pcm_bytes).decode()
    }

    print(f"\nPushing {duration_s:.2f}s of audio ({len(pcm_bytes)} bytes) to {url}...")
    t_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload, timeout=30)
        elapsed = time.perf_counter() - t_start

        if resp.status_code == 200:
            data = resp.json()
            print(f"  OK! duration={data.get('duration_s')}s")
            print(f"  Push took {elapsed:.2f}s (speed={duration_s/elapsed:.1f}x realtime)")
        else:
            print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"  Connection error: {e}")

    # Check status
    try:
        status = requests.get(f"http://{host}:{port}/status", timeout=5).json()
        print(f"\nService status: fps={status.get('fps')}, buffer={status.get('buffer_ms')}ms, "
              f"segment={status.get('current_segment')}, frame={status.get('frame_index')}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push WAV audio to realtime service")
    parser.add_argument("wav_file", help="Path to WAV audio file")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()
    push_wav(args.wav_file, args.host, args.port)
