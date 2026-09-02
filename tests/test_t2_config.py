"""Tests for config module."""
import pytest
import warnings
from pathlib import Path
from realtime_serve.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_normal_creation(self):
        """正常情况：Config 成功创建对象，属性返回正确值"""
        config = Config(character_dir="test_data/wy")
        assert config.character_dir == "test_data/wy"
        assert config.target_fps == 25
        assert config.cap_num == 0

    def test_empty_required_field(self):
        """异常情况：必填字段为空时抛出 TypeError"""
        with pytest.raises(TypeError):
            Config(character_dir="")  # 空路径应该报错

    def test_nonexistent_path_warning(self):
        """边界情况：传入不存在的路径时打印警告但不报错"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = Config(character_dir="/nonexistent/path")
            # 应该产生警告
            assert len(w) >= 1
            assert "does not exist" in str(w[0].message)

    def test_custom_values(self):
        """正常情况：使用自定义值创建 Config"""
        config = Config(
            character_dir="custom/character",
            target_fps=30,
            cap_num=1
        )
        assert config.character_dir == "custom/character"
        assert config.target_fps == 30
        assert config.cap_num == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])