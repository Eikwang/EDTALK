"""Main entry point for realtime streaming inference service.

Usage:
    python -m realtime_serve.main --character-dir test_data/wy \
        --target-fps 25 \
        --cap-num 0
"""
import argparse
import signal
import sys
import threading
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
from realtime_serve.api.app import create_app as create_api_app


# Global state (shared between API server and render loop)
model_manager: ModelManager = None
render_loop: RenderLoop = None
unity_capture: UnityCaptureSender = None
audio_buffer: AudioBuffer = None
audio_processor: AudioProcessor = None
compositor: MouthCompositor = None
face_repair: FaceRepair = None
audio_player: AudioPlayer = None
character_data: CharacterDataPack = None
api_app: FastAPI = None


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
        help="Target frame rate for rendering loop (default: 25)"
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
        help="Disable UnityCapture (for testing without virtual camera)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Frames per preload batch (default: 100, window = 2 × batch_size)"
    )
    parser.add_argument(
        "--no-face-repair",
        action="store_true",
        help="Disable face repair post-processing"
    )
    parser.add_argument(
        "--gaze-correction",
        action="store_true",
        help="Enable gaze correction (placeholder)"
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
    """Initialize all service components.

    Args:
        config: Configuration object
        no_unitycapture: If True, skip UnityCapture initialization

    Returns:
        Tuple of (model_manager, render_loop, unity_capture, audio_buffer, audio_processor, compositor, character_data)
    """
    print("=" * 60)
    print("EDTalk Realtime Streaming Inference Service")
    print("=" * 60)

    # 1. Load character data
    print(f"\n[1/6] Loading character data: {config.character_dir}...")
    character_data = CharacterDataPack(
        config.character_dir,
        batch_size=config.batch_size
    )

    # Get model paths from character directory
    model_paths = character_data.get_model_paths()
    audio2lip_path = model_paths['audio2lip']
    generator_path = model_paths['generator']

    # 2. Load models
    print("\n[2/6] Loading models...")
    model_manager = ModelManager(
        generator_path,
        audio2lip_path,
        device='cuda'
    )
    model_manager.load()

    # 3. Initialize audio buffer and processor
    print("\n[3/6] Initializing audio processing...")
    # max_buffer_seconds=0 means unlimited — supports full-segment push
    audio_buffer = AudioBuffer(max_buffer_seconds=0)
    audio_processor = AudioProcessor()

    # 4. Initialize face repair (conditional)
    face_repair = None
    if config.face_repair_enabled:
        print("\n[4/7] Initializing face repair...")
        face_repair = FaceRepair(model_path='realtime_serve/weights/facefix256.onnx',
                                 alpha=0.7, use_cuda=True)
    else:
        print("\n[4/7] Face repair disabled")

    # 5. Initialize compositor
    print("\n[5/7] Initializing mouth compositor...")
    compositor = MouthCompositor(blur_px=7, landmark_smooth=0.3)

    # Link compositor to character data for mask precomputation
    character_data.set_compositor(compositor)

    # 6. Initialize UnityCapture sender (optional)
    print("\n[6/7] Initializing UnityCapture sender...")
    unity_capture = None
    if not no_unitycapture:
        try:
            # Get resolution from character data
            # resolution is (H, W) from numpy shape[:2]
            resolution = character_data.resolution
            height, width = resolution

            print(f"Character resolution: {resolution} -> Camera: {width}x{height}")

            # Use UnityCapture DirectShow filter
            # Note: Some versions of UnityCapture have issues with certain resolutions
            # Try smaller resolution first for testing
            unity_capture = UnityCaptureSender(
                width=width,
                height=height,
                fps=config.target_fps,
                backend='unitycapture'
            )
        except Exception as e:
            print(f"Warning: Could not initialize UnityCapture: {e}")
            print("Service will run without virtual camera output.")
            print("Use --no-unitycapture to suppress this warning.")
    else:
        print("UnityCapture disabled by --no-unitycapture flag.")

    # 7. Initialize audio player for frame-locked output
    print("\n[7/7] Initializing audio player...")
    # Frame-locked mode: audio is pushed per video frame (640 samples/frame).
    # No pre-buffer needed — audio and video advance together.
    audio_player = AudioPlayer(sample_rate=16000, enabled=True)

    # Create render loop
    print("\nCreating render loop...")
    render_loop = RenderLoop(target_fps=config.target_fps)
    render_loop.set_character_data(character_data)

    # Inject components into render loop
    render_loop.model_manager = model_manager
    render_loop.unity_capture = unity_capture
    render_loop.audio_buffer = audio_buffer
    render_loop.audio_processor = audio_processor
    render_loop.compositor = compositor
    render_loop.face_repair = face_repair
    render_loop.audio_player = audio_player

    print("\nService initialized successfully!")
    print("-" * 60)

    return model_manager, render_loop, unity_capture, audio_buffer, audio_processor, compositor, face_repair, audio_player, character_data


def main():
    """Main entry point."""
    args = parse_args()
    config = Config(
        character_dir=args.character_dir,
        target_fps=args.target_fps,
        cap_num=args.cap_num,
        batch_size=args.batch_size,
        face_repair_enabled=not args.no_face_repair,
        gaze_correction_enabled=args.gaze_correction
    )

    # Setup service components
    global model_manager, render_loop, unity_capture, audio_buffer, audio_processor, compositor, face_repair, audio_player, character_data, api_app

    try:
        result = setup_service(config, no_unitycapture=args.no_unitycapture)
        model_manager, render_loop, unity_capture, audio_buffer, audio_processor, compositor, face_repair, audio_player, character_data = result
    except Exception as e:
        print(f"\nError initializing service: {e}")
        sys.exit(1)

    # Create API app with shared state
    api_app = create_api_app()

    # Inject global state into API module
    import realtime_serve.api.app as api_module
    api_module.audio_buffer = audio_buffer
    api_module.render_loop = render_loop
    api_module.character_data = character_data

    # Start render loop in background thread
    print("\nStarting render loop...")
    render_loop.start()

    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutting down service...")
        render_loop.stop()
        if audio_player:
            audio_player.close()
        if unity_capture:
            unity_capture.close()
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

    uvicorn.run(api_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()