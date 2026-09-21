"""Tests for CharacterDataPack module."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from realtime_serve.sequence_frame_pack import CharacterDataPack


class TestCharacterDataPack:
    """Tests for CharacterDataPack class."""

    def test_normal_load(self, tmp_path):
        """正常情况：成功加载角色数据，发现多个片段"""
        character_dir = tmp_path / "test_character"

        # Create mock directory structure with multiple segments
        orifra_base = character_dir / "orifra"
        count_base = character_dir / "count"
        metadata_base = character_dir / "metadata"
        facemark_base = character_dir / "facemark"
        checkpoint_dir = character_dir / "checkpoint"

        orifra_base.mkdir(parents=True)
        count_base.mkdir(parents=True)
        metadata_base.mkdir(parents=True)
        facemark_base.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)

        # Create two segments: "1" and "happy"
        for segment in ["1", "happy"]:
            orifra_dir = orifra_base / segment
            crop_dir = count_base / segment
            orifra_dir.mkdir()
            crop_dir.mkdir()

            # Create mock frames (3 frames per segment)
            for i in range(3):
                (orifra_dir / f"{i}.jpg").write_bytes(b"fake_image")
                (crop_dir / f"{i}.jpg").write_bytes(b"fake_cropped")

            # Create metadata and facemark
            metadata_path = metadata_base / f"{segment}.npy"
            facemark_path = facemark_base / f"{segment}.npy"

            metadata_dtype = np.dtype([
                ('frame_idx', 'i4'),
                ('crop_x', 'f4'),
                ('crop_y', 'f4'),
                ('crop_size', 'f4'),
            ])
            metadata = np.zeros(3, dtype=metadata_dtype)
            np.save(metadata_path, metadata)

            facemark = np.random.rand(3, 68, 2).astype(np.float32)
            np.save(facemark_path, facemark)

        # Create mock checkpoint files
        (checkpoint_dir / "audio2lip.pt").write_bytes(b"fake_model")
        (checkpoint_dir / "new.pt").write_bytes(b"fake_model")

        with patch('realtime_serve.sequence_frame_pack.cv2.imread') as mock_imread:
            mock_imread.return_value = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

            pack = CharacterDataPack(str(character_dir))

            assert pack.current_segment == "1"  # First segment alphabetically
            assert "1" in pack.segments
            assert "happy" in pack.segments
            assert pack.total_frames == 3

    def test_get_frame_with_index(self, tmp_path):
        """正常情况：get_frame(0) 返回四元组"""
        character_dir = tmp_path / "test_character"

        orifra_base = character_dir / "orifra"
        count_base = character_dir / "count"
        metadata_base = character_dir / "metadata"
        facemark_base = character_dir / "facemark"

        orifra_base.mkdir(parents=True)
        count_base.mkdir(parents=True)
        metadata_base.mkdir(parents=True)
        facemark_base.mkdir(parents=True)

        segment = "1"
        orifra_dir = orifra_base / segment
        crop_dir = count_base / segment
        orifra_dir.mkdir()
        crop_dir.mkdir()

        for i in range(2):
            (orifra_dir / f"{i}.jpg").write_bytes(b"fake")
            (crop_dir / f"{i}.jpg").write_bytes(b"fake")

        metadata_path = metadata_base / f"{segment}.npy"
        facemark_path = facemark_base / f"{segment}.npy"

        metadata_dtype = np.dtype([
            ('frame_idx', 'i4'),
            ('crop_x', 'f4'),
            ('crop_y', 'f4'),
            ('crop_size', 'f4'),
        ])
        metadata = np.zeros(2, dtype=metadata_dtype)
        np.save(metadata_path, metadata)

        facemark = np.random.rand(2, 68, 2).astype(np.float32)
        np.save(facemark_path, facemark)

        with patch('realtime_serve.sequence_frame_pack.cv2.imread') as mock_imread:
            mock_imread.return_value = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

            pack = CharacterDataPack(str(character_dir))
            orifra, cropped, meta, facemark, mask = pack.get_frame(0)

            assert orifra is not None
            assert cropped is not None
            assert meta is not None
            assert facemark is not None
            assert facemark.shape == (68, 2)

    def test_switch_segment(self, tmp_path):
        """正常情况：切换到另一个片段"""
        character_dir = tmp_path / "test_character"

        orifra_base = character_dir / "orifra"
        count_base = character_dir / "count"
        metadata_base = character_dir / "metadata"
        facemark_base = character_dir / "facemark"

        orifra_base.mkdir(parents=True)
        count_base.mkdir(parents=True)
        metadata_base.mkdir(parents=True)
        facemark_base.mkdir(parents=True)

        # Create two segments with different frame counts
        for segment, num_frames in [("1", 3), ("happy", 5)]:
            orifra_dir = orifra_base / segment
            crop_dir = count_base / segment
            orifra_dir.mkdir()
            crop_dir.mkdir()

            for i in range(num_frames):
                (orifra_dir / f"{i}.jpg").write_bytes(b"fake")
                (crop_dir / f"{i}.jpg").write_bytes(b"fake")

            metadata_path = metadata_base / f"{segment}.npy"
            facemark_path = facemark_base / f"{segment}.npy"

            metadata_dtype = np.dtype([
                ('frame_idx', 'i4'),
                ('crop_x', 'f4'),
                ('crop_y', 'f4'),
                ('crop_size', 'f4'),
            ])
            metadata = np.zeros(num_frames, dtype=metadata_dtype)
            np.save(metadata_path, metadata)

            facemark = np.random.rand(num_frames, 68, 2).astype(np.float32)
            np.save(facemark_path, facemark)

        with patch('realtime_serve.sequence_frame_pack.cv2.imread') as mock_imread:
            mock_imread.return_value = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

            pack = CharacterDataPack(str(character_dir))

            # Switch to "happy" segment
            pack.switch_segment("happy")
            assert pack.current_segment == "happy"
            assert pack.total_frames == 5

    def test_nonexistent_path(self):
        """异常情况：路径不存在时抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            CharacterDataPack("/nonexistent/path")

    def test_no_segments(self, tmp_path):
        """异常情况：没有可用片段时抛出 ValueError"""
        character_dir = tmp_path / "test_character"

        orifra_base = character_dir / "orifra"
        orifra_base.mkdir(parents=True)

        with pytest.raises(ValueError, match="No segments found"):
            CharacterDataPack(str(character_dir))

    def test_invalid_segment_switch(self, tmp_path):
        """异常情况：切换到不存在的片段抛出 ValueError"""
        character_dir = tmp_path / "test_character"

        orifra_base = character_dir / "orifra"
        count_base = character_dir / "count"
        metadata_base = character_dir / "metadata"
        facemark_base = character_dir / "facemark"

        orifra_base.mkdir(parents=True)
        count_base.mkdir(parents=True)
        metadata_base.mkdir(parents=True)
        facemark_base.mkdir(parents=True)

        segment = "1"
        orifra_dir = orifra_base / segment
        crop_dir = count_base / segment
        orifra_dir.mkdir()
        crop_dir.mkdir()

        for i in range(2):
            (orifra_dir / f"{i}.jpg").write_bytes(b"fake")
            (crop_dir / f"{i}.jpg").write_bytes(b"fake")

        metadata_path = metadata_base / f"{segment}.npy"
        facemark_path = facemark_base / f"{segment}.npy"

        metadata_dtype = np.dtype([
            ('frame_idx', 'i4'),
            ('crop_x', 'f4'),
            ('crop_y', 'f4'),
            ('crop_size', 'f4'),
        ])
        metadata = np.zeros(2, dtype=metadata_dtype)
        np.save(metadata_path, metadata)

        facemark = np.random.rand(2, 68, 2).astype(np.float32)
        np.save(facemark_path, facemark)

        with patch('realtime_serve.sequence_frame_pack.cv2.imread') as mock_imread:
            mock_imread.return_value = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

            pack = CharacterDataPack(str(character_dir))

            with pytest.raises(ValueError, match="not found"):
                pack.switch_segment("nonexistent")

    def test_get_model_paths(self, tmp_path):
        """正常情况：获取模型路径"""
        character_dir = tmp_path / "test_character"

        orifra_base = character_dir / "orifra"
        count_base = character_dir / "count"
        metadata_base = character_dir / "metadata"
        facemark_base = character_dir / "facemark"
        checkpoint_dir = character_dir / "checkpoint"

        orifra_base.mkdir(parents=True)
        count_base.mkdir(parents=True)
        metadata_base.mkdir(parents=True)
        facemark_base.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)

        segment = "1"
        orifra_dir = orifra_base / segment
        crop_dir = count_base / segment
        orifra_dir.mkdir()
        crop_dir.mkdir()

        for i in range(2):
            (orifra_dir / f"{i}.jpg").write_bytes(b"fake")
            (crop_dir / f"{i}.jpg").write_bytes(b"fake")

        metadata_path = metadata_base / f"{segment}.npy"
        facemark_path = facemark_base / f"{segment}.npy"

        metadata_dtype = np.dtype([
            ('frame_idx', 'i4'),
            ('crop_x', 'f4'),
            ('crop_y', 'f4'),
            ('crop_size', 'f4'),
        ])
        metadata = np.zeros(2, dtype=metadata_dtype)
        np.save(metadata_path, metadata)

        facemark = np.random.rand(2, 68, 2).astype(np.float32)
        np.save(facemark_path, facemark)

        # Create mock checkpoint files
        (checkpoint_dir / "audio2lip.pt").write_bytes(b"fake_model")
        (checkpoint_dir / "new.pt").write_bytes(b"fake_model")

        with patch('realtime_serve.sequence_frame_pack.cv2.imread') as mock_imread:
            mock_imread.return_value = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

            pack = CharacterDataPack(str(character_dir))

            paths = pack.get_model_paths()
            assert paths['audio2lip'].endswith("audio2lip.pt")
            assert paths['generator'].endswith("new.pt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])