"""Main entry point for realtime streaming inference service.

Usage:
    python -m realtime_serve.main --character-dir test_data/wy \
        --target-fps 25 \
        --cap-num 0

Important semantic change:
    --target-fps now controls ONLY the final output frame rate to the
    virtual camera. The render loop produces frames as fast as the GPU
    allows and pushes them into a bounded output buffer. This decouples
    output pacing from rendering jitter.
"""
import argparse
import signal
import sys
import threading
import time
import ctypes
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from realtime_serve.config import Config
from realtime_serve.models import ModelManager
from realtime_serve.sequence_frame_pack import CharacterDataPack
from realtime_serve.render_loop import RenderLoop
from realtime_serve.unity_capture import UnityCaptureSender
from realtime_serve.audio_buffer import AudioBuffer
from realtime_serve.audio_processor import AudioProcessor
from realtime_serve.compositor import MouthCompositor
from realtime_serve.face_repair import FaceRepair
from realtime_serve.audio_player import AudioPlayer
from realtime_serve.output_buffer import FrameBuffer, OutputStreamer
from realtime_serve.api.app import create_app as create_api_app


# Global state (shared between API server and render loop)
model_manager: ModelManager = None
render_loop: RenderLoop = None
output_streamer: OutputStreamer = None
frame_buffer: FrameBuffer = None
unity_capture: UnityCaptureSender = None
audio_buffer: AudioBuffer = None
audio_processor: AudioProcessor = None
compositor: MouthCompositor = None
face_repair: FaceRepair = None
audio_player: AudioPlayer = None
character_data: CharacterDataPack = None
api_app: FastAPI = None


def _set_timer_resolution(enabled: bool):
    """Set Windows timer resolution to 1ms for tighter sleep precision."""
    try:
        if enabled:
            ctypes.windll.winmm.timeBeginPeriod(1)
        else:
            ctypes.windll.winmm.timeEndPeriod(1)
    except Exception as e:
        print(f"[main] Warning: could not set Windows timer resolution: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Realtime Streaming Inference Service for EDTalk"
    )
    parser.add_argument(
        "--character-dir",
        type=str,
        default="test_data/wy",
        help="Path to character directory containing all data (default: test_data/wy)"
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=25,
        help="Target output frame rate to the virtual camera (default: 25). "
             "NOTE: this now paces only camera output, not rendering."
    )
    parser.add_argument(
        "--cap-num",
        type=int,
        default=0,
        help="UnityCapture camera number (default: 0)"
    )
    parser.add_argument(
        "--no-unitycapture",
        action="store_true",
        help="Disable UnityCapture (for testing without virtual camera). "
             "Output pacing still runs; frames are dropped after accounting."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Frames per preload batch (default: 100, window = 2 × batch_size). "
             "100 frames = 4s of content at 25fps, which the preloader can "
             "load in ~0.5s with parallel decode. Larger batches make the "
             "preload fall behind and freeze frames at batch boundaries."
    )
    parser.add_argument(
        "--no-face-repair",
        action="store_true",
        help="Disable face repair post-processing (default: off)"
    )
    parser.add_argument(
        "--face-repair",
        choices=["on", "adaptive"],
        default=None,
        help="Face repair mode. 'on': repair every frame (~9ms/frame "
             "synchronous GPU wait). 'adaptive': repair only while the "
             "output buffer has headroom (recommended when quality matters)."
    )
    parser.add_argument(
        "--gaze-correction",
        action="store_true",
        help="Enable gaze correction (placeholder)"
    )
    parser.add_argument(
        "--gen-fp16",
        action="store_true",
        help="Enable Generator fp16 autocast inference. Off by default: "
             "measured slower than fp32+TF32 on RTX 4080 SUPER."
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=16,
        help="Output frame buffer capacity in frames (default: 16, max: 300). "
             "Each frame is ~11MB at 1440x2560. Deep buffers absorb larger "
             "GPU jitter but increase memory and startup latency."
    )
    parser.add_argument(
        "--preroll-frames",
        type=int,
        default=8,
        help="Frames required in the buffer before camera output starts "
             "(default: 8, must be <= --buffer-frames). Larger values trade "
             "startup latency for smoother startup."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    return parser.parse_args()


def setup_service(config: Config, no_unitycapture: bool = False):
    """Initialize all service components."""
    print("=" * 60)
    print("EDTalk Realtime Streaming Inference Service")
    print("=" * 60)

    memory_mb = config.buffer_frames * 11  # rough per-frame BGR bytes / 1MB
    preroll_s = config.preroll_frames / max(1, config.target_fps)
    print(f"\n[Config] target_fps={config.target_fps} output pacing only")
    print(f"[Config] buffer_frames={config.buffer_frames} (~{memory_mb}MB)")
    print(f"[Config] preroll_frames={config.preroll_frames} (~{preroll_s:.2f}s startup delay)")

    # 1. Load character data
    print(f"\n[1/7] Loading character data: {config.character_dir}...")
    character_data = CharacterDataPack(
        config.character_dir,
        batch_size=config.batch_size
    )

    model_paths = character_data.get_model_paths()
    audio2lip_path = model_paths['audio2lip']
    generator_path = model_paths['generator']

    # 2. Load models
    print("\n[2/7] Loading models...")
    model_manager = ModelManager(
        generator_path,
        audio2lip_path,
        device='cuda',
        use_fp16=config.gen_fp16
    )
    model_manager.load()

    # 3. Initialize audio buffer and processor
    print("\n[3/7] Initializing audio processing...")
    audio_buffer = AudioBuffer(max_buffer_seconds=0)
    audio_processor = AudioProcessor()

    # 4. Initialize face repair (conditional)
    face_repair = None
    if config.face_repair_mode in ('on', 'adaptive'):
        print(f"\n[4/7] Initializing face repair (mode: {config.face_repair_mode})...")
        face_repair = FaceRepair(model_path='realtime_serve/weights/facefix256.onnx',
                                 alpha=0.7, use_cuda=True)
        face_repair.warmup()
    else:
        print("\n[4/7] Face repair disabled")

    # 5. Initialize compositor
    print("\n[5/7] Initializing mouth compositor...")
    compositor = MouthCompositor(blur_px=7, landmark_smooth=0.3)
    character_data.set_compositor(compositor)

    # 6. Initialize output buffer and streamer
    print("\n[6/7] Initializing output buffer and streamer...")
    frame_buffer = FrameBuffer(
        capacity=config.buffer_frames,
        reserve=min(5, config.buffer_frames)
    )

    # 7. Initialize UnityCapture sender (optional)
    print("\n[7/7] Initializing UnityCapture sender...")
    unity_capture = None
    sender_fn = None
    if not no_unitycapture:
        try:
            resolution = character_data.resolution
            height, width = resolution
            print(f"Character resolution: {resolution} -> Camera: {width}x{height}")
            unity_capture = UnityCaptureSender(
                width=width,
                height=height,
                fps=config.target_fps,
                backend='unitycapture'
            )
            sender_fn = unity_capture.send
        except Exception as e:
            print(f"Warning: Could not initialize UnityCapture: {e}")
            print("Service will run without virtual camera output.")
            print("Use --no-unitycapture to suppress this warning.")
    else:
        print("UnityCapture disabled by --no-unitycapture flag.")

    # Initialize audio player
    print("\nInitializing audio player...")
    audio_player = AudioPlayer(sample_rate=16000, enabled=True)

    # Create render loop
    print("\nCreating render loop...")
    render_loop = RenderLoop(target_fps=config.target_fps)
    render_loop.set_character_data(character_data)

    # Create output streamer
    output_streamer = OutputStreamer(
        frame_buffer=frame_buffer,
        target_fps=config.target_fps,
        sender=sender_fn,
        audio_start_callback=None,  # wired below after audio_player exists
        preroll_frames=config.preroll_frames
    )

    # Inject components into render loop
    render_loop.model_manager = model_manager
    render_loop.frame_buffer = frame_buffer
    render_loop.output_streamer = output_streamer
    render_loop.audio_buffer = audio_buffer
    render_loop.audio_processor = audio_processor
    render_loop.compositor = compositor
    render_loop.face_repair = face_repair
    render_loop.face_repair_mode = config.face_repair_mode
    render_loop.audio_player = audio_player

    # Audio-start callback: triggered by OutputStreamer at display time.
    def _audio_start_callback():
        pcm = None
        with render_loop._pending_lock:
            pcm = render_loop._pending_pcm
            render_loop._pending_pcm = None
        if pcm is not None and audio_player is not None:
            audio_player.reset()
            audio_player.push_pcm(pcm)
            with render_loop._audio_lock:
                render_loop._audio_start_time = time.perf_counter()
            print("[OutputStreamer] Audio playback started at display time")

    output_streamer.audio_start_callback = _audio_start_callback

    print("\nService initialized successfully!")
    print("-" * 60)

    return (model_manager, render_loop, output_streamer, frame_buffer,
            unity_capture, audio_buffer, audio_processor, compositor,
            face_repair, audio_player, character_data)


def main():
    """Main entry point."""
    args = parse_args()
    # Face repair mode resolution: --no-face-repair wins (back-compat alias
    # for off); otherwise --face-repair on|adaptive; otherwise off.
    if args.no_face_repair:
        repair_mode = 'off'
    elif args.face_repair is not None:
        repair_mode = args.face_repair
    else:
        repair_mode = 'off'

    config = Config(
        character_dir=args.character_dir,
        target_fps=args.target_fps,
        cap_num=args.cap_num,
        batch_size=args.batch_size,
        face_repair_enabled=(repair_mode != 'off'),
        face_repair_mode=repair_mode,
        gaze_correction_enabled=args.gaze_correction,
        buffer_frames=args.buffer_frames,
        preroll_frames=args.preroll_frames,
        gen_fp16=args.gen_fp16
    )

    # Set Windows timer resolution early.
    _set_timer_resolution(True)

    # Setup service components
    global model_manager, render_loop, output_streamer, frame_buffer
    global unity_capture, audio_buffer, audio_processor, compositor
    global face_repair, audio_player, character_data, api_app

    try:
        result = setup_service(config, no_unitycapture=args.no_unitycapture)
        (model_manager, render_loop, output_streamer, frame_buffer,
         unity_capture, audio_buffer, audio_processor, compositor,
         face_repair, audio_player, character_data) = result
    except Exception as e:
        print(f"\nError initializing service: {e}")
        _set_timer_resolution(False)
        sys.exit(1)

    # Create API app with shared state
    api_app = create_api_app()

    # Inject global state into API module
    import realtime_serve.api.app as api_module
    api_module.audio_buffer = audio_buffer
    api_module.render_loop = render_loop
    api_module.output_streamer = output_streamer
    api_module.frame_buffer = frame_buffer
    api_module.character_data = character_data

    # Start output streamer before render loop so it owns pacing immediately.
    print("\nStarting output streamer...")
    output_streamer.start()

    # Start render loop in background thread
    print("Starting render loop...")
    render_loop.start()

    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutting down service...")
        render_loop.stop()
        if output_streamer is not None:
            output_streamer.stop()
        if audio_player is not None:
            audio_player.close()
        if unity_capture is not None:
            unity_capture.close()
        _set_timer_resolution(False)
        print("Service stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start API server (blocks until interrupted)
    print(f"\nAPI server running at http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  POST /audio/push      - Push audio chunk (streaming TTS)")
    print(f"  POST /audio/push_full - Push full audio segment (file-based)")
    print(f"  POST /segment/switch  - Switch segment")
    print(f"  GET  /status          - Get service status")
    print("\nPress Ctrl+C to stop.\n")

    try:
        uvicorn.run(api_app, host=args.host, port=args.port)
    finally:
        _set_timer_resolution(False)


if __name__ == "__main__":
    main()
