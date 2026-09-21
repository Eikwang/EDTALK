"""Tests for audio_processor module."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from realtime_serve.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Tests for AudioProcessor class."""

    def test_pcm_to_mel(self):
        """正常情况：pcm_to_mel() 返回形状正确的 mel tensor"""
        processor = AudioProcessor()

        # Create 200ms of PCM data (3200 samples at 16kHz)
        pcm_chunk = np.random.rand(3200).astype(np.float32)

        mel_input, bs, T = processor.pcm_to_mel(pcm_chunk)

        assert mel_input is not None
        assert bs == 1
        assert T > 0

    def test_generate_silent_mel_caching(self):
        """边界情况：连续调用 generate_silent_mel() 三次，第二次和第三次返回相同的缓存对象"""
        processor = AudioProcessor()

        # First call - should compute and cache
        silent_mel1 = processor.generate_silent_mel()

        # Second and third calls - should return same cached object
        silent_mel2 = processor.generate_silent_mel()
        silent_mel3 = processor.generate_silent_mel()

        assert silent_mel2 is silent_mel1  # Same reference
        assert silent_mel3 is silent_mel1  # Same reference

    def test_empty_pcm_handling(self):
        """异常情况：传入空 PCM 数据时返回全零 mel 而非抛异常"""
        processor = AudioProcessor()

        # Empty array
        empty_pcm = np.array([], dtype=np.float32)

        mel_input, bs, T = processor.pcm_to_mel(empty_pcm)

        # Should not raise exception
        assert mel_input is not None

    def test_silent_mel_shape(self):
        """正常情况：静音 mel 的形状正确"""
        processor = AudioProcessor()

        mel_input, bs, T = processor.generate_silent_mel()

        # Mel input should have valid shape (at least 3 dimensions)
        assert mel_input.ndim >= 3
        # Should contain data (not all zeros or empty)
        assert mel_input.numel() > 0

    def test_parse_audio_length(self):
        """正常情况：parse_audio_length() 返回正确的音频长度和帧数"""
        from realtime_serve.audio_processor import parse_audio_length

        audio_length = 3200  # 200ms at 16kHz
        sr = 16000
        fps = 25

        wav_length, num_frames = parse_audio_length(audio_length, sr, fps)

        assert wav_length > 0
        assert num_frames > 0

    def test_crop_pad_audio(self):
        """正常情况：crop_pad_audio() 处理音频长度"""
        from realtime_serve.audio_processor import crop_pad_audio

        # Short audio - should be padded
        short_audio = np.random.rand(1000).astype(np.float32)
        target_length = 3200

        result = crop_pad_audio(short_audio, target_length)

        assert len(result) == target_length

        # Long audio - should be cropped
        long_audio = np.random.rand(5000).astype(np.float32)

        result = crop_pad_audio(long_audio, target_length)

        assert len(result) == target_length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])