"""Configuration management for realtime streaming inference service."""
from dataclasses import dataclass
import warnings
from pathlib import Path


@dataclass
class Config:
    """Configuration class for the inference service.

    Holds all configuration parameters including model paths, character directory,
    and rendering settings.
    """

    character_dir: str = "test_data/wy"
    target_fps: int = 25
    cap_num: int = 0
    batch_size: int = 100  # frames per preload batch (window = 2 × batch_size)
    face_repair_enabled: bool = True  # enable face repair post-processing
    gaze_correction_enabled: bool = False  # enable gaze correction (placeholder)

    def __post_init__(self):
        """Validate paths after initialization."""
        # Check if character directory exists (warn but don't fail)
        if not Path(self.character_dir).exists():
            warnings.warn(
                f"Character directory does not exist: {self.character_dir}",
                UserWarning,
                stacklevel=2
            )

        # Validate required fields
        if not self.character_dir:
            raise TypeError("character_dir is required")