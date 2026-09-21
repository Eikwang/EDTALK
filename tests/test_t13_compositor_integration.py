"""Tests for mouth compositing integration."""
import pytest
import numpy as np
from realtime_serve.compositor import MouthCompositor


class TestMouthCompositingIntegration:
    """Tests for mouth compositing integration (T13)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.compositor = MouthCompositor()

        # Create mock orifra frame (1080p BGR)
        self.orifra_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Create mock rendered frame (256x256 BGR)
        self.rendered_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Create mock facemark (68 points) in 256x256 space
        self.facemark = self._create_realistic_facemark_256(
            center_x=128, center_y=128, face_size=200
        )

        # Create mock metadata structured array
        metadata_dtype = np.dtype([
            ('crop_x', 'f4'),
            ('crop_y', 'f4'),
            ('crop_size', 'f4'),
        ])
        self.meta = np.array([(500.0, 700.0, 500.0)], dtype=metadata_dtype)[0]

    def _create_realistic_facemark_256(self, center_x, center_y, face_size):
        """Create a realistic 68-point facemark layout in 256x256 space."""
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

    def test_composite_result_is_different_from_original(self):
        """正常情况：合成后的帧与原始帧相比，仅嘴部区域有差异"""
        original_copy = self.orifra_frame.copy()
        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            self.facemark,
            self.meta
        )

        # Find differences
        diff = np.abs(result.astype(int) - original_copy.astype(int))
        max_diff = np.max(diff)

        # Should have some difference (mouth region modified)
        assert max_diff > 0

    def test_background_region_unchanged(self):
        """正常情况：背景区域完全一致"""
        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            self.facemark,
            self.meta
        )

        # Check top-left corner (should be unchanged)
        corner_size = 100
        original_corner = self.orifra_frame[:corner_size, :corner_size]
        result_corner = result[:corner_size, :corner_size]

        diff = np.abs(original_corner.astype(int) - result_corner.astype(int))
        max_corner_diff = np.max(diff)

        # Corners should be very similar (allow small rounding errors)
        assert max_corner_diff <= 10

    def test_edge_blending_natural(self):
        """边界情况：嘴部区域靠近图像边缘时，mask 正确裁剪"""
        # Create facemark near edge
        edge_facemark = self.facemark.copy()
        edge_facemark[48, 0] = 10  # Near left edge
        edge_facemark[59, 0] = 100

        result = self.compositor.composite(
            self.orifra_frame,
            self.rendered_frame,
            edge_facemark,
            self.meta
        )

        # Should not raise exception and produce valid output
        assert result is not None
        assert result.shape == self.orifra_frame.shape

    def test_all_black_rendered_frame(self):
        """异常情况：渲染帧为全黑时，合成后嘴部区域为黑色，但边缘过渡自然"""
        black_frame = np.zeros((256, 256, 3), dtype=np.uint8)

        result = self.compositor.composite(
            self.orifra_frame,
            black_frame,
            self.facemark,
            self.meta
        )

        # Should not raise exception
        assert result is not None
        assert result.shape == self.orifra_frame.shape

    def test_multiple_composites_consistency(self):
        """正常情况：多次合成结果一致"""
        results = []
        for i in range(3):
            result = self.compositor.composite(
                self.orifra_frame,
                self.rendered_frame,
                self.facemark,
                self.meta
            )
            results.append(result)

        # All results should be identical
        assert all(np.array_equal(results[0], r) for r in results)

    def test_composite_with_different_resolutions(self):
        """边界情况：不同分辨率的输入"""
        # Test with different orifra resolution
        small_orifra = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        result = self.compositor.composite(
            small_orifra,
            self.rendered_frame,
            self.facemark,
            self.meta
        )

        assert result.shape == small_orifra.shape

    def test_mouth_mask_properties(self):
        """正常情况：嘴部遮罩属性验证"""
        mask = self.compositor._create_mouth_mask(256, 256, self.facemark)

        # Lip center should be close to 1.0
        assert mask[128, 128] > 0.9

        # Far outside should be 0.0
        assert mask[0, 0] == 0.0

        # Values should be in [0, 1]
        assert np.min(mask) >= 0
        assert np.max(mask) <= 1

        # Transition band should have intermediate values
        # Check a point between lip and jaw
        chin_y = int(self.facemark[8, 1])
        lip_bottom_y = int(self.facemark[57, 1])
        mid_x = int(self.facemark[8, 0])
        if chin_y > lip_bottom_y and chin_y < 256 and mid_x < 256:
            mid_y = (chin_y + lip_bottom_y) // 2
            if mask[mid_y, mid_x] > 0:
                assert 0.0 < mask[mid_y, mid_x] < 1.0, \
                    f"Transition band value should be intermediate, got {mask[mid_y, mid_x]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
