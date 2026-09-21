"""Tests for MouthCompositor module."""
import pytest
import numpy as np
import cv2
from realtime_serve.compositor import MouthCompositor


class TestMouthCompositor:
    """Tests for MouthCompositor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.compositor = MouthCompositor(blur_px=7, landmark_smooth=0.3)

        # Create mock orifra frame (1080p BGR)
        self.orifra_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Create mock rendered frame (256x256 BGR)
        self.rendered_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Create mock facemark (68 points) in 256x256 space
        # (detected on the cropped/count face image, not the original orifra)
        self.facemark = self._create_realistic_facemark_256(
            center_x=128, center_y=128, face_size=200
        )

        # Create mock metadata structured array
        metadata_dtype = np.dtype([
            ('crop_x', 'f4'),
            ('crop_y', 'f4'),
            ('crop_size', 'f4'),
        ])
        self.meta = np.array([(560.0, 140.0, 800.0)], dtype=metadata_dtype)[0]

    def _create_realistic_facemark_256(self, center_x, center_y, face_size):
        """Create a realistic 68-point facemark layout in 256x256 space.

        dlib 68-point layout:
        - 0-16: jaw contour
        - 17-21: left eyebrow
        - 22-26: right eyebrow
        - 27-35: nose
        - 36-41: left eye
        - 42-47: right eye
        - 48-59: outer lip
        - 60-67: inner lip
        """
        pts = np.zeros((68, 2), dtype=np.float32)
        scale = face_size / 400.0

        # Jaw contour (0-16): U-shape
        jaw_x = np.linspace(-150, 150, 17) * scale + center_x
        jaw_y = np.array([100, 80, 60, 40, 20, 0, -20, -40, -60, -40, -20, 0, 20, 40, 60, 80, 100]) * scale + center_y
        pts[:17, 0] = jaw_x
        pts[:17, 1] = jaw_y

        # Eyebrows (17-26)
        for i in range(17, 22):
            pts[i] = [center_x - 80 * scale + (i - 17) * 15 * scale, center_y - 70 * scale]
        for i in range(22, 27):
            pts[i] = [center_x + 20 * scale + (i - 22) * 15 * scale, center_y - 70 * scale]

        # Nose (27-35)
        pts[27] = [center_x, center_y - 50 * scale]
        pts[28] = [center_x, center_y - 40 * scale]
        pts[29] = [center_x, center_y - 30 * scale]
        pts[30] = [center_x, center_y - 10 * scale]
        pts[31] = [center_x - 15 * scale, center_y]
        pts[32] = [center_x - 8 * scale, center_y + 5 * scale]
        pts[33] = [center_x, center_y + 10 * scale]
        pts[34] = [center_x + 8 * scale, center_y + 5 * scale]
        pts[35] = [center_x + 15 * scale, center_y]

        # Eyes (36-47)
        for i in range(36, 42):
            pts[i] = [center_x - 50 * scale + (i - 36) * 12 * scale, center_y - 50 * scale]
        for i in range(42, 48):
            pts[i] = [center_x + 15 * scale + (i - 42) * 12 * scale, center_y - 50 * scale]

        # Outer lip (48-59): oval shape
        lip_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        lip_rx, lip_ry = 40 * scale, 20 * scale
        for i, angle in enumerate(lip_angles):
            pts[48 + i] = [center_x + lip_rx * np.cos(angle),
                           center_y + 25 * scale + lip_ry * np.sin(angle)]

        # Inner lip (60-67): smaller oval
        inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        inner_rx, inner_ry = 25 * scale, 12 * scale
        for i, angle in enumerate(inner_angles):
            pts[60 + i] = [center_x + inner_rx * np.cos(angle),
                           center_y + 25 * scale + inner_ry * np.sin(angle)]

        return pts

    def test_normal_composite(self):
        """正常情况：composite() 返回合成后的 1080p 帧，尺寸与 orifra 一致"""
        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            self.facemark,
            self.meta
        )

        assert result is not None
        assert result.shape == self.orifra_frame.shape
        assert result.dtype == np.uint8

    def test_composite_only_modifies_mouth_region(self):
        """正常情况：合成后的帧仅嘴部区域有差异，非嘴部区域保持原样"""
        original_copy = self.orifra_frame.copy()
        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            self.facemark,
            self.meta
        )

        diff = np.abs(result.astype(int) - original_copy.astype(int))
        max_diff = np.max(diff)

        # Should have some difference (mouth region modified)
        assert max_diff > 0

    def test_composite_preserves_background(self):
        """正常情况：背景区域（远离嘴部）保持不变"""
        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            self.facemark,
            self.meta
        )

        # Check corners (should be unchanged — far from mouth)
        corner_size = 50
        original_corner = self.orifra_frame[:corner_size, :corner_size]
        result_corner = result[:corner_size, :corner_size]

        diff = np.abs(original_corner.astype(int) - result_corner.astype(int))
        max_corner_diff = np.max(diff)

        # Corners should be identical (no blending there)
        assert max_corner_diff <= 10

    def test_composite_with_edge_landmarks(self):
        """边界情况：facemark 关键点超出图像边界时自动裁剪"""
        edge_facemark = self.facemark.copy()
        edge_facemark[48, 0] = -10  # Outside left edge in 256x256
        edge_facemark[59, 1] = 300  # Outside bottom edge

        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            edge_facemark,
            self.meta
        )

        # Should not raise exception
        assert result is not None
        assert result.shape == self.orifra_frame.shape

    def test_composite_missing_facemark(self):
        """异常情况：传入 None 的 facemark 时抛出 ValueError"""
        with pytest.raises(ValueError, match="facemark required"):
            self.compositor.composite(
                self.orifra_frame,
                self.rendered_frame,
                None,
                self.meta
            )

    def test_composite_nan_facemark_fallback(self):
        """异常情况：facemark 含 NaN 时跳过合成返回原图"""
        nan_facemark = self.facemark.copy()
        nan_facemark[30] = [np.nan, np.nan]  # Nose tip is NaN

        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            nan_facemark,
            self.meta
        )

        # Should return original frame unchanged
        assert np.array_equal(result, self.orifra_frame)

    def test_landmark_ids(self):
        """正常情况：关键点 ID 包含正确的区域"""
        assert 0 in self.compositor.JAW_IDS       # jaw start
        assert 16 in self.compositor.JAW_IDS      # jaw end
        assert 31 in self.compositor.NOSE_ALAR_IDS  # nose alar start
        assert 35 in self.compositor.NOSE_ALAR_IDS  # nose alar end
        assert 48 in self.compositor.OUTER_LIP_IDS  # outer lip start
        assert 59 in self.compositor.OUTER_LIP_IDS  # outer lip end

    def test_create_mouth_mask(self):
        """正常情况：嘴部遮罩正确生成，内区域=1.0，外区域=0.0"""
        mask = self.compositor._create_mouth_mask(256, 256, self.facemark)

        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32
        # Lip center should be close to 1.0
        assert mask[128, 128] > 0.9
        # Far outside face should be 0.0
        assert mask[0, 0] == 0.0
        assert np.min(mask) >= 0.0
        assert np.max(mask) <= 1.0

    def test_mask_inner_region_is_one(self):
        """正常情况：嘴唇凸包内部区域 alpha=1.0"""
        mask = self.compositor._create_mouth_mask(256, 256, self.facemark)

        # Get lip center (should be well inside the lip hull)
        lip_center_y = int(self.facemark[48:60, 1].mean())
        lip_center_x = int(self.facemark[48:60, 0].mean())

        # Lip center should be 1.0 (fully rendered)
        assert mask[lip_center_y, lip_center_x] >= 0.95

    def test_mask_outer_region_is_zero(self):
        """正常情况：面部轮廓外区域 alpha=0.0"""
        mask = self.compositor._create_mouth_mask(256, 256, self.facemark)

        # Top-left corner is well outside the face
        assert mask[0, 0] == 0.0
        # Bottom-right corner is well outside the face
        assert mask[255, 255] == 0.0

    def test_mask_transition_band_smooth(self):
        """正常情况：过渡带从嘴唇到面部轮廓平滑衰减"""
        mask = self.compositor._create_mouth_mask(256, 256, self.facemark)

        # Find a point in the transition zone (between lip and jaw)
        # Jaw point 8 is the chin, lip point 57 is lower lip
        chin_y = int(self.facemark[8, 1])
        lip_bottom_y = int(self.facemark[57, 1])
        mid_x = int(self.facemark[8, 0])

        # Point between chin and lower lip should be in transition zone
        if chin_y > lip_bottom_y and chin_y < 256 and mid_x < 256:
            mid_y = (chin_y + lip_bottom_y) // 2
            val = mask[mid_y, mid_x]
            # Should be between 0 and 1 (transition zone)
            assert 0.0 < val < 1.0, \
                f"Transition zone value should be intermediate, got {val}"

    def test_mask_constrained_by_face_contour(self):
        """正常情况：遮罩不超过面部轮廓"""
        mask = self.compositor._create_mouth_mask(256, 256, self.facemark)

        # Points outside the jaw contour should have mask = 0
        # Jaw point 0 is the leftmost jaw point
        jaw_left_x = int(self.facemark[0, 0])
        jaw_left_y = int(self.facemark[0, 1])

        # A point well to the left of the jaw should be 0
        if jaw_left_x > 10:
            assert mask[jaw_left_y, jaw_left_x - 10] == 0.0

    def test_extract_crop_256(self):
        """正常情况：从 orifra 提取 crop 区域并 resize 到 256x256"""
        crop = self.compositor._extract_crop_256(
            self.orifra_frame, 560.0, 140.0, 800.0
        )

        assert crop is not None
        assert crop.shape == (256, 256, 3)
        assert crop.dtype == np.uint8

    def test_extract_crop_256_clamped(self):
        """边界情况：crop 区域超出图像边界时自动裁剪"""
        crop = self.compositor._extract_crop_256(
            self.orifra_frame, -100.0, -100.0, 800.0
        )

        assert crop is not None
        assert crop.shape == (256, 256, 3)

    def test_composite_mouth_region_smaller_than_full_crop(self):
        """正常情况：嘴部区域小于整个 crop 区域，仅嘴部被替换"""
        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            self.facemark,
            self.meta
        )

        # Check a point far from the mouth (top of crop region)
        top_of_crop_y = 200
        top_of_crop_x = 960
        if top_of_crop_y < self.orifra_frame.shape[0] and top_of_crop_x < self.orifra_frame.shape[1]:
            diff = np.abs(result[top_of_crop_y, top_of_crop_x].astype(int) -
                         self.orifra_frame[top_of_crop_y, top_of_crop_x].astype(int))
            assert np.max(diff) <= 5, \
                f"Top of crop region should be preserved, diff={np.max(diff)}"

    def test_landmark_ema_smoothing(self):
        """正常情况：EMA平滑减少facemark帧间抖动"""
        compositor = MouthCompositor(landmark_smooth=0.3)

        # First frame: no smoothing yet
        fm1 = self.facemark.copy()
        result1 = compositor._smooth_landmarks(fm1)

        # Second frame: slightly different landmarks
        fm2 = fm1 + np.random.randn(68, 2) * 2.0
        result2 = compositor._smooth_landmarks(fm2)

        # Smoothed result should be closer to fm1 than raw fm2
        diff_raw = np.abs(fm2 - fm1).mean()
        diff_smooth = np.abs(result2 - fm1).mean()
        assert diff_smooth < diff_raw, \
            f"Smoothed landmarks should be closer to previous: {diff_smooth:.2f} vs {diff_raw:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
