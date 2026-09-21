"""Tests for audio-driven streaming integration."""
import pytest
import numpy as np
import base64
from unittest.mock import MagicMock, patch
from realtime_serve.audio_buffer import AudioBuffer
from realtime_serve.audio_processor import AudioProcessor


class TestAudioDrivenIntegration:
    """Tests for audio-driven streaming integration (T11)."""

    def test_audio_push_and_consume(self):
        """正常情况：推送音频后可以消费"""
        buffer = AudioBuffer()
        processor = AudioProcessor()

        # Push 200ms of audio
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data)

        # Consume chunk
        chunk = buffer.consume_chunk()
        assert chunk is not None
        assert len(chunk) == 3200

        # Convert to mel
        mel_input, bs, T = processor.pcm_to_mel(chunk)
        assert mel_input is not None

    def test_silent_mode_when_no_audio(self):
        """边界情况：停止推送音频后，使用静音模式"""
        buffer = AudioBuffer()
        processor = AudioProcessor()

        # No audio pushed
        chunk = buffer.consume_chunk()
        assert chunk is None

        # Should use silent mel
        silent_mel, bs, T = processor.generate_silent_mel()
        assert silent_mel is not None

    def test_continuous_audio_streaming(self):
        """正常情况：持续推送音频，缓冲区不溢出"""
        buffer = AudioBuffer()

        # Simulate continuous streaming
        for i in range(50):
            pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
            buffer.push(pcm_data)

            # Consume some chunks
            if i % 5 == 0:
                buffer.consume_chunk()

        # Buffer should not exceed 2 seconds
        duration_ms = buffer.buffer_duration_ms()
        assert duration_ms <= 2100

    def test_rapid_audio_push(self):
        """异常情况：快速连续推送 100 次（每次 40ms）后，缓冲区不溢出"""
        buffer = AudioBuffer()

        # Push 100 chunks of 40ms each (total 4 seconds)
        for i in range(100):
            pcm_data = (np.random.rand(640) * 32767).astype(np.int16).tobytes()
            buffer.push(pcm_data)

        # Buffer should still be limited to 2 seconds
        duration_ms = buffer.buffer_duration_ms()
        assert duration_ms <= 2100

    def test_audio_processing_pipeline(self):
        """正常情况：完整音频处理流水线"""
        buffer = AudioBuffer()
        processor = AudioProcessor()

        # Push audio
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data)

        # Consume and process
        chunk = buffer.consume_chunk()
        if chunk is not None:
            mel_input, bs, T = processor.pcm_to_mel(chunk)
            assert mel_input.numel() > 0

    def test_buffer_overflow_handling(self):
        """边界情况：音频推送速度 > 消费速度时，丢弃最早数据"""
        buffer = AudioBuffer(max_buffer_seconds=2.0)

        # Fill buffer beyond max with sequence numbers
        for i in range(10):
            pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
            buffer.push(pcm_data, seq=i)

        # Buffer should be at max size
        duration_ms = buffer.buffer_duration_ms()
        assert duration_ms >= 1900  # Close to 2 seconds

    def test_end_to_end_latency(self):
        """正常情况：端到端延迟测量"""
        import time

        buffer = AudioBuffer()
        processor = AudioProcessor()

        # Measure time from push to mel generation
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()

        t_start = time.perf_counter()
        buffer.push(pcm_data)
        chunk = buffer.consume_chunk()
        mel_input, bs, T = processor.pcm_to_mel(chunk)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        assert latency_ms < 100  # Should be well under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])