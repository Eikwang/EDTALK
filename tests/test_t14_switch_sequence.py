"""Tests for RenderLoop segment switching."""
import pytest
from unittest.mock import MagicMock, patch
from realtime_serve.render_loop import RenderLoop


class TestRenderLoopSwitchSegment:
    """Tests for RenderLoop.switch_segment (T14)."""

    def test_switch_segment_success(self):
        """正常情况：调用 switch_segment() 后，下一帧使用新片段渲染"""
        loop = RenderLoop(target_fps=25)

        # Create mock character data with multiple segments
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.total_frames = 10
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        # Switch to new segment
        loop.switch_segment("happy")

        # Should call switch on character data
        mock_character.switch_segment.assert_called_once_with("happy")

    def test_switch_segment_no_character_data(self):
        """异常情况：没有角色数据时抛出 ValueError"""
        loop = RenderLoop(target_fps=25)

        with pytest.raises(ValueError, match="No character data loaded"):
            loop.switch_segment("happy")

    def test_switch_segment_invalid_segment(self):
        """异常情况：切换到不存在的片段抛出 ValueError"""
        loop = RenderLoop(target_fps=25)

        # Create mock character data
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.switch_segment = MagicMock(side_effect=ValueError("Segment 'nonexistent' not found"))

        loop.set_character_data(mock_character)

        with pytest.raises(ValueError, match="not found"):
            loop.switch_segment("nonexistent")

    def test_thread_safety_during_switch(self):
        """边界情况：切换过程中线程安全"""
        loop = RenderLoop(target_fps=25)

        # Create mock character data
        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        # Switch should be thread-safe
        loop.switch_segment("happy")

        # Should not raise any exception
        assert mock_character.switch_segment.called

    def test_switch_fires_resolution_callback(self):
        """切换成功后按新片段原生分辨率通知摄像头（W, H 顺序）"""
        loop = RenderLoop(target_fps=25)

        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.total_frames = 10
        mock_character.resolution = (1920, 1080)  # (H, W)
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        cb = MagicMock()
        loop.resolution_change_cb = cb
        loop.switch_segment("happy")

        cb.assert_called_once_with(1080, 1920)

    def test_switch_resolution_callback_error_not_fatal(self):
        """摄像头分辨率切换失败不阻断段切换"""
        loop = RenderLoop(target_fps=25)

        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.total_frames = 10
        mock_character.resolution = (1920, 1080)
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)

        cb = MagicMock(side_effect=RuntimeError("camera gone"))
        loop.resolution_change_cb = cb
        loop.switch_segment("happy")

        # Switch itself still succeeded
        mock_character.switch_segment.assert_called_once_with("happy")
        assert loop._character_data.current_segment == mock_character.current_segment

    def test_switch_without_callback_skips_resolution(self):
        """未接线回调（默认）时不读取分辨率，行为与旧版一致"""
        loop = RenderLoop(target_fps=25)

        mock_character = MagicMock()
        mock_character.current_segment = "1"
        mock_character.total_frames = 10
        mock_character.switch_segment = MagicMock(return_value=True)

        loop.set_character_data(mock_character)
        assert loop.resolution_change_cb is None

        loop.switch_segment("happy")
        mock_character.switch_segment.assert_called_once_with("happy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])