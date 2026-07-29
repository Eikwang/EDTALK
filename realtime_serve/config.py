"""Configuration management for realtime streaming inference service."""
from dataclasses import dataclass
import warnings
from pathlib import Path


@dataclass
class Config:
    """Configuration class for the inference service.

    Holds all configuration parameters including model paths, dataset paths,
    and rendering settings.
    """

    model_path: str = "ckpts/EDTalk_lip_pose.pt"
    audio2lip_path: str = "ckpts/Audio2Lip.pt"
    default_dataset_dir: str = "HDTF/subject_001"
    default_video_name: str = "video_clip_1"
    target_fps: int = 25
    cap_num: int = 0

    def __post_init__(self):
        """Validate paths after initialization."""
        # Check if model files exist (warn but don't fail)
        optional_paths = [
            ("model_path", self.model_path),
            ("audio2lip_path", self.audio2lip_path),
        ]

        for name, path in optional_paths:
            if not Path(path).exists():
                warnings.warn(
                    f"{name} does not exist: {path}",
                    UserWarning,
                    stacklevel=2
                )

        # Validate required fields
        if not self.model_path:
            raise TypeError("model_path is required")
        if not self.audio2lip_path:
            raise TypeError("audio2lip_path is required")
