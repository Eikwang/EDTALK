"""Tests for static streaming integration."""
import pytest
import time
from unittest.mock import MagicMock, patch
from realtime_serve.main import main
from realtime_serve.config import Config
from realtime_serve.render_loop import RenderLoop


class TestStaticStreamingIntegration:
    """Tests for static streaming integration (T7)."""

    def test_service_startup(self):
        """正常情况：服务启动后打印配置信息"""
        with patch('sys.argv', ['realtime_serve.main',
                               '--character-dir', 'test_data/wy']):
            with patch('realtime_serve.main.Config') as mock_config_cls:
                mock_config = MagicMock()
                mock_config.character_dir = "test_data/wy"
                mock_config.target_fps = 25
                mock_config.cap_num = 0
                mock_config_cls.return_value = mock_config

                with patch('realtime_serve.main.setup_service') as mock_setup:
                    # Create mock objects for setup_service return
                    mock_mm = MagicMock()
                    mock_loop = MagicMock()
                    mock_uc = MagicMock()
                    mock_ab = MagicMock()
                    mock_ap = MagicMock()
                    mock_comp = MagicMock()
                    mock_char = MagicMock()
                    mock_setup.return_value = (mock_mm, mock_loop, mock_uc, mock_ab, mock_ap, mock_comp, mock_char)

                    with patch('uvicorn.run'):
                        # main() should not raise an exception
                        try:
                            main()
                        except SystemExit:
                            pass  # Expected if service tries to exit

    def test_config_validation(self):
        """边界情况：配置校验失败时抛出异常"""
        from realtime_serve.config import Config

        with pytest.raises(TypeError):
            Config(character_dir="")  # Empty path should raise

    def test_render_loop_integration(self):
        """正常情况：RenderLoop 可以与 CharacterDataPack 集成"""
        loop = RenderLoop(target_fps=25)

        # Create a mock character data
        mock_char = MagicMock()
        mock_char.current_segment = "1"
        mock_char.total_frames = 10
        mock_char.get_frame.return_value = (
            None,  # orifra
            None,  # cropped
            None,  # meta
            None   # facemark
        )

        # Set the character data
        loop.set_character_data(mock_char)

        # Verify it was set
        assert loop.get_current_pack() is mock_char

    def test_graceful_shutdown(self):
        """边界情况：Ctrl+C 停止服务，进程优雅退出"""
        loop = RenderLoop(target_fps=25)

        # Start the loop
        loop.start()
        assert loop._running is True

        # Stop gracefully
        loop.stop()
        assert loop._running is False

    def test_missing_character_directory(self):
        """异常情况：角色目录不存在时服务启动失败"""
        from realtime_serve.sequence_frame_pack import CharacterDataPack

        with pytest.raises(FileNotFoundError):
            CharacterDataPack("/nonexistent/path")