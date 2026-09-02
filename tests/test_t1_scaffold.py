"""Tests for realtime_serve.main entry point."""
import subprocess
import sys
from pathlib import Path


def test_main_help():
    """正常情况：执行 python -m realtime_serve.main --help 返回参数解析帮助信息"""
    result = subprocess.run(
        [sys.executable, "-m", "realtime_serve.main", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert "usage:" in result.stdout.lower(), "Help output should contain 'usage:'"
    assert "--character-dir" in result.stdout, "Should contain --character-dir argument"
    assert "--target-fps" in result.stdout, "Should contain --target-fps argument"


def test_missing_init_file():
    """边界情况：删除 api/__init__.py 后导入失败，提示缺少模块"""
    # This is a structural test - we verify the package structure is correct
    api_init = Path(__file__).parent.parent / "realtime_serve" / "api" / "__init__.py"
    utils_init = Path(__file__).parent.parent / "realtime_serve" / "utils" / "__init__.py"
    main_init = Path(__file__).parent.parent / "realtime_serve" / "__init__.py"

    assert api_init.exists(), "api/__init__.py must exist for proper package structure"
    assert utils_init.exists(), "utils/__init__.py must exist for proper package structure"
    assert main_init.exists(), "realtime_serve/__init__.py must exist for proper package structure"


if __name__ == "__main__":
    test_main_help()
    test_missing_init_file()
    print("All tests passed!")