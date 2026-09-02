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
    target_fps: int = 25  # now controls only output pacing, not production
    cap_num: int = 0
    batch_size: int = 100  # frames per preload batch (window = 2 × batch_size)
    face_repair_enabled: bool = True  # enable face repair post-processing
    face_repair_mode: str = 'off'  # 'off' | 'on' | 'adaptive' (repair only with buffer headroom)
    gaze_correction_enabled: bool = False  # enable gaze correction (placeholder)
    buffer_frames: int = 16  # output frame buffer capacity (frames, ~11MB each)
    preroll_frames: int = 8  # frames required before output starts
    # Generator fp16 autocast: DISABLED by default. Measured negative on
    # RTX 4080 SUPER (gen p50 21-34ms fp32 -> constant 36ms fp16): with
    # TF32 enabled, Ada gets most of the fp16 throughput already, so
    # autocast only adds per-layer cast overhead.
    gen_fp16: bool = False

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

        if self.face_repair_mode not in ('off', 'on', 'adaptive'):
            raise ValueError(
                f"face_repair_mode must be 'off', 'on' or 'adaptive', "
                f"got '{self.face_repair_mode}'"
            )

        if self.buffer_frames < 1:
            raise ValueError(f"buffer_frames must be >= 1, got {self.buffer_frames}")
        if self.buffer_frames > 300:
            raise ValueError(f"buffer_frames must be <= 300, got {self.buffer_frames}")
        if self.preroll_frames < 0:
            raise ValueError(f"preroll_frames must be >= 0, got {self.preroll_frames}")
        if self.preroll_frames > self.buffer_frames:
            raise ValueError(
                f"preroll_frames ({self.preroll_frames}) must be <= "
                f"buffer_frames ({self.buffer_frames})"
            )