"""Main entry point for realtime streaming inference service."""
import argparse
import sys
from realtime_serve.config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Realtime Streaming Inference Service for EDTalk"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ckpts/EDTalk_lip_pose.pt",
        help="Path to Generator model weights (default: ckpts/EDTalk_lip_pose.pt)"
    )
    parser.add_argument(
        "--audio2lip-path",
        type=str,
        default="ckpts/Audio2Lip.pt",
        help="Path to Audio2Lip model weights (default: ckpts/Audio2Lip.pt)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="HDTF/subject_001",
        help="Default dataset directory containing sequence frames (default: HDTF/subject_001)"
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="video_clip_1",
        help="Default video name in dataset (default: video_clip_1)"
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
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config from arguments
    config = Config(
        model_path=args.model_path,
        audio2lip_path=args.audio2lip_path,
        default_dataset_dir=args.dataset_dir,
        default_video_name=args.video_name,
        target_fps=args.target_fps,
        cap_num=args.cap_num
    )

    print(f"EDTalk Realtime Streaming Inference Service")
    print(f"Model path: {config.model_path}")
    print(f"Audio2Lip path: {config.audio2lip_path}")
    print(f"Dataset: {config.default_dataset_dir}/{config.default_video_name}")
    print(f"Target FPS: {config.target_fps}")
    print(f"Camera number: {config.cap_num}")
    print("\nService initialization not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
