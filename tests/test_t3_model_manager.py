"""Tests for ModelManager module."""
import pytest
import torch
from unittest.mock import MagicMock, patch
from realtime_serve.models import ModelManager


class TestModelManager:
    """Tests for ModelManager class."""

    def test_normal_load(self):
        """正常情况：load() 后 audio2lip 和 gen 均为 GPU tensor"""
        with patch('torch.load') as mock_load, \
             patch('realtime_serve.models.Audio2Lip') as mock_audio2lip_cls, \
             patch('realtime_serve.models.Generator') as mock_gen_cls:

            # Mock model instances
            mock_audio2lip = MagicMock()
            mock_audio2lip.to.return_value = mock_audio2lip
            mock_audio2lip_cls.return_value = mock_audio2lip

            mock_gen = MagicMock()
            mock_gen.to.return_value = mock_gen
            mock_gen_cls.return_value = mock_gen

            # Mock weights
            mock_load.side_effect = [
                {"audio2lip": {"layer1": torch.randn(10)}},
                {"gen": {"layer1": torch.randn(10)}}
            ]

            manager = ModelManager("fake_model.pt", "fake_audio2lip.pt")
            manager.load()

            assert manager.audio2lip is not None
            assert manager.gen is not None
            mock_audio2lip.load_state_dict.assert_called_once()
            mock_gen.load_state_dict.assert_called_once()

    def test_corrupted_model_file(self):
        """边界情况：模型文件损坏时捕获异常并打印错误信息"""
        with patch('torch.load', side_effect=RuntimeError("Corrupted file")):
            manager = ModelManager("corrupted.pt", "corrupted_audio2lip.pt")
            with pytest.raises(RuntimeError, match="Corrupted file"):
                manager.load()

    def test_cuda_unavailable(self):
        """异常情况：CUDA 不可用时抛出 RuntimeError"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('realtime_serve.models.Audio2Lip') as mock_cls:

            manager = ModelManager("fake_model.pt", "fake_audio2lip.pt")
            with pytest.raises(RuntimeError, match="CUDA"):
                manager.load()

    def test_infer_lip(self):
        """正常情况：infer_lip() 返回 lip 系数"""
        manager = ModelManager("fake_model.pt", "fake_audio2lip.pt")
        manager.audio2lip = MagicMock()
        manager.audio2lip.return_value = [torch.randn(1, 20)]

        mel_input = torch.randn(1, 1, 80, 16)
        bs, T = 1, 5

        with patch('realtime_serve.models.conv_feat') as mock_conv:
            mock_conv.return_value = torch.randn(1, 20)
            result = manager.infer_lip(mel_input, bs, T)

        assert result.shape == (1, 20)

    def test_infer_frame(self):
        """正常情况：infer_frame() 返回渲染帧"""
        manager = ModelManager("fake_model.pt", "fake_audio2lip.pt")
        manager.gen = MagicMock()
        manager.gen.test_from_audio_pose_image.return_value = torch.randn(1, 3, 256, 256)

        img_source = torch.randn(1, 3, 256, 256)
        alpha_D_lip = torch.randn(1, 20)
        pose_img_drive = torch.randn(1, 3, 256, 256)

        result = manager.infer_frame(img_source, alpha_D_lip, pose_img_drive)

        assert result.shape == (1, 3, 256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
