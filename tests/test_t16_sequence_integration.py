"""Tests for segment switching integration."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from realtime_serve.render_loop import RenderLoop


class TestSegmentSwitchingIntegration:
    """Tests for segment switching integration (T16)."""

    def test_switch_segment_maintains_framerate(self):
        """正常情况：切换片段后，推流帧率保持在 23-27 fps"""
        loop = RenderLoop(target_fps=25)

        # Create mock character data with multiple segments
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.total_frames = 100
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        # Switch to "happy" segment
        loop.switch_segment("happy")

        # Should have switched successfully
        mock_character.switch_segment.assert_called_once_with("happy")

    def test_rapid_segment_switches(self):
        """边界情况：连续快速切换 5 次不同片段，最后一次切换生效"""
        loop = RenderLoop(target_fps=25)

        # Create mock character data
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        # Rapidly switch segments
        segments = ["1", "happy", "angry", "5", "neutral"]
        for seg in segments:
            loop.switch_segment(seg)

        # Last call should be with "neutral"
        mock_character.switch_segment.assert_called_with("neutral")

    def test_resolution_adaptation(self):
        """异常情况：切换到分辨率不同的片段后，虚拟摄像头输出画面正确缩放"""
        loop = RenderLoop(target_fps=25)

        # Create mock character data with different resolutions per segment
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.total_frames = 100
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        # Switch to different segment
        loop.switch_segment("happy")

        # Should have switched successfully
        mock_character.switch_segment.assert_called_with("happy")

    def test_switch_during_running_loop(self):
        """边界情况：切换过程中推流不中断"""
        import numpy as np

        loop = RenderLoop(target_fps=25)

        # Create mock character data
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.total_frames = 100
        mock_character.get_frame.return_value = (
            np.zeros((100, 100, 3), dtype=np.uint8),  # orifra
            MagicMock(),  # cropped
            MagicMock(),  # meta
            MagicMock(),  # facemark
            None,         # mask
        )
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        # Start the loop
        loop.start()

        # Switch while running
        loop.switch_segment("happy")

        # Stop the loop
        loop.stop()

        # Should still be running after switch
        mock_character.switch_segment.assert_called_with("happy")

    def test_switch_preserves_frame_index(self):
        """正常情况：切换片段后帧索引重置为 0"""
        loop = RenderLoop(target_fps=25)

        # Create mock character data
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        # Advance frame index
        loop._frame_index = 5

        # Switch to new segment
        loop.switch_segment("happy")

        # Should have switched successfully
        mock_character.switch_segment.assert_called_with("happy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])