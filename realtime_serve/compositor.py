"""Mouth compositor for extracting mouth region and compositing onto original frame.

Uses facemark 68-point landmarks (in 256x256 space) to define a precise mouth
region, creates a landmark-guided mask, and composites only the mouth area from
the rendered frame onto the original frame.

Mask strategy:
  - Inner region (mouth): outer lip landmarks 48-59 → convex hull → alpha=1.0
  - Outer boundary: jaw contour 0-16 (left/right/bottom) + nose alar 31-35 (top)
  - Transition: from mouth hull outward to the outer boundary, smooth falloff
  - Constraint: mask never exceeds the face contour or goes above the nose alar

This avoids the background deformation artifact that occurs when the entire 256x256
rendered face (which has distorted background outside the face) is pasted back.

IMPORTANT: facemark coordinates are expected to be in 256x256 space (i.e.,
detected on the cropped/count face image, not the original orifra frame).
This is set during preprocessing in crop_head_video.py, which detects landmarks
on the 256x256 resized crop rather than the original frame. This avoids
coordinate mapping errors and ensures pixel-accurate mask alignment.
"""
import numpy as np
import cv2


class MouthCompositor:
    """Compositor that extracts mouth region from rendered frame and composites onto original frame.

    Mask strategy — landmark-guided dual-boundary mask:
    1. Inner boundary: convex hull of outer lip landmarks (48-59) → alpha = 1.0
    2. Outer boundary: jaw contour (0-16) + nose alar (31-35) → alpha = 0.0
    3. Between inner and outer: distance-based smooth falloff
    4. EMA smoothing on landmarks to prevent edge flickering

    The mask is constrained to never exceed the face contour (jaw) or go above
    the nose alar line, ensuring only the mouth region is composited.
    """

    # dlib 68-point landmark indices
    # Jaw contour: 0-16 (face outline, used as outer boundary)
    # Nose alar: 31-35 (nose bottom, used as upper boundary)
    # Outer lip: 48-59 (mouth outline, used as inner boundary = alpha=1.0)
    # Inner lip: 60-67 (not used for mask, but part of mouth)
    JAW_IDS = list(range(0, 17))           # face contour
    NOSE_ALAR_IDS = list(range(31, 36))    # nose bottom / alar
    OUTER_LIP_IDS = list(range(48, 60))    # outer lip contour

    def __init__(self, blur_px: int = 7, landmark_smooth: float = 0.3):
        """Initialize MouthCompositor.

        Args:
            blur_px: Gaussian blur kernel size for the final mask (must be odd).
                     Provides sub-pixel smoothness to eliminate any remaining hard edges.
            landmark_smooth: EMA smoothing factor for facemark landmarks (0-1).
                             0 = no smoothing (use raw landmarks each frame)
                             0.3 = moderate smoothing (default, good anti-flicker)
                             1.0 = frozen (never updates from first frame)
        """
        self.blur_px = blur_px if blur_px % 2 == 1 else blur_px + 1
        self.landmark_smooth = landmark_smooth

        # EMA-smoothed facemark (in 256x256 space), persists across frames
        self._smoothed_facemark: np.ndarray | None = None

        # Mask cache: avoid recomputing expensive distance transforms every frame
        # The mask changes slowly (landmarks shift ~2-5px between frames at 25fps),
        # so we only recompute when landmarks move beyond a tolerance threshold.
        self._cached_mask: np.ndarray | None = None
        self._cached_mask_facemark: np.ndarray | None = None
        self._mask_cache_tolerance: float = 3.0  # pixels — recompute if any landmark moves more than this

    def composite(self, orifra_frame: np.ndarray, rendered_frame_256: np.ndarray,
                  facemark: np.ndarray, meta: np.ndarray) -> np.ndarray:
        """Composite mouth region from rendered frame onto original frame.

        Optimized: operates only on the crop region, not the full frame.
        Avoids the expensive full-frame copy (10.8MB at 1440x2560).

        Args:
            orifra_frame: Original 1080p frame (H, W, 3) BGR uint8
            rendered_frame_256: Rendered frame (256, 256, 3) BGR uint8
            facemark: (68, 2) float32 landmark points in 256x256 space
                      (detected on the cropped/count face image)
            meta: Single metadata entry with crop_x, crop_y, crop_size fields

        Returns:
            Composited 1080p frame (H, W, 3) BGR uint8.
            NOTE: The crop region of orifra_frame is modified in-place.
                  This is safe because the preloaded buffer is replaced
                  before any frame is reused.

        Raises:
            ValueError: If facemark is None
        """
        if facemark is None:
            raise ValueError("facemark required")

        # Get crop parameters from metadata
        crop_x = float(meta['crop_x'])
        crop_y = float(meta['crop_y'])
        crop_size = float(meta['crop_size'])

        if crop_size <= 0:
            return orifra_frame

        # facemark is already in 256x256 space — no coordinate mapping needed
        facemark_256 = facemark

        # Check for NaN landmarks (face not detected in that frame)
        if np.any(np.isnan(facemark_256)):
            return orifra_frame

        # --- EMA smooth facemark to prevent edge flickering ---
        facemark_256 = self._smooth_landmarks(facemark_256)

        # --- Create landmark-guided mask (cached — recompute only when landmarks shift) ---
        if self._cached_mask is not None and self._cached_mask_facemark is not None:
            max_shift = np.max(np.abs(facemark_256 - self._cached_mask_facemark))
            if max_shift < self._mask_cache_tolerance:
                mask_256 = self._cached_mask
            else:
                mask_256 = self._create_mouth_mask(256, 256, facemark_256)
                self._cached_mask = mask_256
                self._cached_mask_facemark = facemark_256.copy()
        else:
            mask_256 = self._create_mouth_mask(256, 256, facemark_256)
            self._cached_mask = mask_256
            self._cached_mask_facemark = facemark_256.copy()

        # --- Resize rendered and mask to orifra crop scale ---
        crop_size_int = int(round(crop_size))
        if crop_size_int <= 0:
            return orifra_frame

        rendered_resized = cv2.resize(rendered_frame_256, (crop_size_int, crop_size_int),
                                      interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_256, (crop_size_int, crop_size_int),
                                  interpolation=cv2.INTER_LINEAR)

        # --- Compute crop region bounds ---
        x1 = int(round(crop_x))
        y1 = int(round(crop_y))
        x2 = x1 + crop_size_int
        y2 = y1 + crop_size_int

        # Clamp to frame boundaries
        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(orifra_frame.shape[1], x2)
        y2_c = min(orifra_frame.shape[0], y2)

        # Corresponding region in resized images
        rx1 = x1_c - x1
        ry1 = y1_c - y1
        rx2 = rx1 + (x2_c - x1_c)
        ry2 = ry1 + (y2_c - y1_c)

        if rx2 <= rx1 or ry2 <= ry1:
            return orifra_frame

        local_mask = mask_resized[ry1:ry2, rx1:rx2]

        # Skip if mask is all zero
        if np.max(local_mask) < 1e-6:
            return orifra_frame

        # Extract only the crop region from orifra (small slice, not full frame copy)
        target_region = orifra_frame[y1_c:y2_c, x1_c:x2_c].copy()
        rendered_region = rendered_resized[ry1:ry2, rx1:rx2]

        # Blend: rendered * mask + orifra * (1 - mask)
        mask_3ch = np.stack([local_mask] * 3, axis=-1)
        blended = (rendered_region.astype(np.float32) * mask_3ch +
                   target_region.astype(np.float32) * (1.0 - mask_3ch))

        # Assign back to crop region of original frame (no full-frame copy needed)
        orifra_frame[y1_c:y2_c, x1_c:x2_c] = blended.astype(np.uint8)

        return orifra_frame

    def _smooth_landmarks(self, facemark_256: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to facemark landmarks to prevent edge flickering.

        Per-frame landmark jitter causes the mask boundary to shift slightly,
        which makes the mask edge jump between frames → visible flickering.
        EMA smoothing stabilizes the boundary position.

        Args:
            facemark_256: Current frame's landmarks (68, 2) in 256x256 space

        Returns:
            Smoothed landmarks (68, 2)
        """
        if self._smoothed_facemark is None:
            # First frame: initialize
            self._smoothed_facemark = facemark_256.copy()
            return facemark_256.copy()

        alpha = self.landmark_smooth
        self._smoothed_facemark = (
            self._smoothed_facemark * (1.0 - alpha) + facemark_256 * alpha
        )
        return self._smoothed_facemark.copy()

    def _create_mouth_mask(self, width: int, height: int,
                           facemark: np.ndarray) -> np.ndarray:
        """Create a landmark-guided mouth mask with smooth falloff.

        The mask is defined by two boundaries:
        - Inner boundary: convex hull of outer lip landmarks (48-59) → alpha = 1.0
        - Outer boundary: jaw contour (0-16) + nose alar (31-35) → alpha = 0.0

        Between the two boundaries, alpha falls off smoothly based on
        normalized distance from the inner boundary to the outer boundary.

        Steps:
        1. Build inner mask: convex hull of outer lip → fillPoly
        2. Build outer mask: convex hull of jaw + nose alar → fillPoly
        3. Compute distance from inner boundary (distanceTransform on ~inner_mask)
        4. Compute distance from outer boundary (distanceTransform on outer_mask)
        5. For pixels between inner and outer: alpha = dist_to_outer / (dist_to_inner + dist_to_outer)
           → 1.0 at inner edge, 0.0 at outer edge, smooth in between
        6. Gaussian blur for sub-pixel smoothness

        Args:
            width: Mask width
            height: Mask height
            facemark: (68, 2) float32 landmarks in 256x256 space (already smoothed)

        Returns:
            Mask (H, W) float32 in range [0, 1]
        """
        # --- Extract landmark groups ---
        lip_pts = facemark[self.OUTER_LIP_IDS].astype(np.float32)   # 48-59
        jaw_pts = facemark[self.JAW_IDS].astype(np.float32)         # 0-16
        nose_pts = facemark[self.NOSE_ALAR_IDS].astype(np.float32)  # 31-35

        # --- Inner boundary: convex hull of outer lip ---
        lip_hull = cv2.convexHull(lip_pts)  # (N, 1, 2)
        inner_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(inner_mask, [lip_hull.astype(np.int32)], 255)

        # --- Outer boundary: convex hull of jaw + nose alar ---
        # This defines the face region that constrains the mask
        outer_pts = np.vstack([jaw_pts, nose_pts]).astype(np.float32)
        outer_hull = cv2.convexHull(outer_pts)  # (N, 1, 2)
        outer_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(outer_mask, [outer_hull.astype(np.int32)], 255)

        # --- Distance transforms ---
        # Distance from inner boundary (for pixels outside inner_mask)
        dist_from_inner = cv2.distanceTransform(
            255 - inner_mask, cv2.DIST_L2, 5
        ).astype(np.float32)

        # Distance from outer boundary (for pixels inside outer_mask)
        dist_from_outer = cv2.distanceTransform(
            outer_mask, cv2.DIST_L2, 5
        ).astype(np.float32)

        # --- Build mask ---
        mask = np.zeros((height, width), dtype=np.float32)

        # Inside lip hull: fully rendered
        mask[inner_mask > 0] = 1.0

        # Transition zone: between inner and outer boundaries
        # Pixels that are outside inner_mask but inside outer_mask
        transition_zone = (inner_mask == 0) & (outer_mask > 0)

        if np.any(transition_zone):
            d_inner = dist_from_inner[transition_zone]
            d_outer = dist_from_outer[transition_zone]
            total = d_inner + d_outer
            # Avoid division by zero
            total = np.maximum(total, 1e-6)
            # Normalized: 1.0 at inner edge, 0.0 at outer edge
            alpha = d_outer / total
            # Apply smoothstep for smoother falloff
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            mask[transition_zone] = alpha

        # --- Gaussian blur for sub-pixel smoothness ---
        if self.blur_px > 1:
            mask = cv2.GaussianBlur(mask, (self.blur_px, self.blur_px), 0)
            mask = np.clip(mask, 0.0, 1.0)

        return mask

    def _extract_crop_256(self, orifra_frame: np.ndarray,
                          crop_x: float, crop_y: float,
                          crop_size: float) -> np.ndarray | None:
        """Extract the crop region from orifra and resize to 256x256.

        Kept for test compatibility.
        """
        x1 = int(round(crop_x))
        y1 = int(round(crop_y))
        x2 = x1 + int(round(crop_size))
        y2 = y1 + int(round(crop_size))

        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(orifra_frame.shape[1], x2)
        y2_c = min(orifra_frame.shape[0], y2)

        if x2_c <= x1_c or y2_c <= y1_c:
            return None

        crop = orifra_frame[y1_c:y2_c, x1_c:x2_c]

        h, w = crop.shape[:2]
        expected = int(round(crop_size))
        if h < expected or w < expected:
            padded = np.zeros((expected, expected, 3), dtype=np.uint8)
            off_y = y1_c - y1
            off_x = x1_c - x1
            padded[off_y:off_y + h, off_x:off_x + w] = crop
            crop = padded

        return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
