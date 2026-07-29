"""Tests for config module."""
import pytest
import warnings
from pathlib import Path
from realtime_serve.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_normal_creation(self):
        """正常情况：Config 成功创建对象，属性返回正确值"""
        config = Config(
            model_path="ckpts/EDTalk_lip_pose.pt",
            audio2lip_path="ckpts/Audio2Lip.pt"
        )
        assert config.model_path == "ckpts/EDTalk_lip_pose.pt"
        assert config.audio2lip_path == "ckpts/Audio2Lip.pt"
        assert config.default_dataset_dir == "HDTF/subject_001"
        assert config.default_video_name == "video_clip_1"
        assert config.target_fps == 25

    def test_empty_required_field(self):
        """异常情况：必填字段为空时抛出 TypeError"""
        with pytest.raises(TypeError):
            Config(model_path="")  # 空路径应该报错

    def test_nonexistent_path_warning(self):
        """边界情况：传入不存在的路径时打印警告但不报错"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = Config(
                model_path="/nonexistent/path/model.pt",
                audio2lip_path="ckpts/Audio2Lip.pt"
            )
            # 应该产生警告
            assert len(w) >= 1
            assert "does not exist" in str(w[0].message)

    def test_custom_values(self):
        """正常情况：使用自定义值创建 Config"""
        config = Config(
            model_path="custom/model.pt",
            audio2lip_path="custom/audio2lip.pt",
            default_dataset_dir="custom/dataset",
            default_video_name="custom_video",
            target_fps=30
        )
        assert config.model_path == "custom/model.pt"
        assert config.audio2lip_path == "custom/audio2lip.pt"
        assert config.default_dataset_dir == "custom/dataset"
        assert config.default_video_name == "custom_video"
        assert config.target_fps == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
