"""Tests for AudioBuffer module."""
import pytest
import numpy as np
import threading
import time
from realtime_serve.audio_buffer import AudioBuffer


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_normal_push(self):
        """正常情况：push() 后 buffer_duration_ms() 返回正值"""
        buffer = AudioBuffer()

        # Create 200ms of PCM data (16kHz, mono, int16)
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data)

        duration_ms = buffer.buffer_duration_ms()
        assert duration_ms > 0
        assert duration_ms >= 190  # Should be around 200ms

    def test_consume_chunk(self):
        """正常情况：consume_chunk() 返回 200ms PCM 数据"""
        buffer = AudioBuffer()

        # Push enough data for one chunk (200ms)
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data)

        chunk = buffer.consume_chunk()
        assert chunk is not None
        assert len(chunk) == 3200  # 200ms at 16kHz

    def test_consume_chunk_empty(self):
        """边界情况：缓冲不足时 consume_chunk() 返回 None"""
        buffer = AudioBuffer()

        # Push less than 200ms
        pcm_data = (np.random.rand(1000) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data)

        chunk = buffer.consume_chunk()
        assert chunk is None

    def test_buffer_max_size(self):
        """边界情况：设置 max_buffer_seconds 后，连续 push 超过限制时长时自动丢弃旧数据"""
        buffer = AudioBuffer(max_buffer_seconds=2.0)

        # Push more than 2 seconds of audio with sequence numbers
        for i in range(10):
            pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
            buffer.push(pcm_data, seq=i)

        duration_ms = buffer.buffer_duration_ms()
        assert duration_ms <= 2100  # Should not exceed 2 seconds significantly

    def test_buffer_unlimited(self):
        """正常情况：默认 max_buffer_seconds=0 时，buffer 可以存储完整音频段"""
        buffer = AudioBuffer()  # default: unlimited

        # Push 10 seconds of audio (50 chunks of 200ms) with sequence numbers
        for i in range(50):
            pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
            buffer.push(pcm_data, seq=i)

        duration_ms = buffer.buffer_duration_ms()
        assert duration_ms >= 9900  # Should be ~10 seconds

    def test_consume_removes_data(self):
        """正常情况：consume_chunk() 移除已消费的数据"""
        buffer = AudioBuffer()

        # Push 400ms of audio
        pcm_data = (np.random.rand(6400) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data)

        # Consume one chunk (200ms)
        chunk = buffer.consume_chunk()
        assert chunk is not None

        # Remaining should be ~200ms
        remaining_duration = buffer.buffer_duration_ms()
        assert 180 <= remaining_duration <= 220

    def test_thread_safety(self):
        """异常情况：多线程并发 push/consume 时无数据竞争错误"""
        buffer = AudioBuffer()
        errors = []

        def producer():
            try:
                for _ in range(100):
                    pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
                    buffer.push(pcm_data)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(50):
                    buffer.consume_chunk()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        # Start threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        # No errors should occur
        assert len(errors) == 0

    def test_buffer_duration_after_consume(self):
        """正常情况：消费后缓冲时长减少"""
        buffer = AudioBuffer()

        # Push 400ms
        pcm_data = (np.random.rand(6400) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data)

        initial_duration = buffer.buffer_duration_ms()

        # Consume 200ms
        buffer.consume_chunk()

        final_duration = buffer.buffer_duration_ms()
        assert final_duration < initial_duration

    def test_push_full_clears_existing(self):
        """正常情况：push_full() 清除已有数据并加载完整段"""
        buffer = AudioBuffer()

        # Push some initial data
        pcm_old = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_old)
        assert buffer.buffer_duration_ms() > 0

        # Push full segment — should replace old data
        pcm_full = (np.random.rand(16000) * 32767).astype(np.int16).tobytes()  # 1 second at 16kHz
        buffer.push_full(pcm_full)

        duration_ms = buffer.buffer_duration_ms()
        assert 950 <= duration_ms <= 1050  # Should be ~1 second, not 1.2s

    def test_push_full_then_consume(self):
        """正常情况：push_full() 后可以逐 chunk 消费"""
        buffer = AudioBuffer()

        # Push 1 second of audio
        pcm_full = (np.random.rand(16000) * 32767).astype(np.int16).tobytes()
        buffer.push_full(pcm_full)

        # Should be able to consume 5 chunks (5 × 200ms = 1s)
        chunks = []
        for _ in range(5):
            chunk = buffer.consume_chunk()
            assert chunk is not None
            chunks.append(chunk)

        # Buffer should be empty now
        chunk = buffer.consume_chunk()
        assert chunk is None

    def test_push_full_resets_sequence(self):
        """正常情况：push_full() 重置序列号状态"""
        buffer = AudioBuffer()

        # Push with sequence numbers
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        buffer.push(pcm_data, seq=0)
        buffer.push(pcm_data, seq=1)

        # push_full should reset sequence state
        pcm_full = (np.random.rand(32000) * 32767).astype(np.int16).tobytes()
        buffer.push_full(pcm_full)

        # Should be able to push with seq=0 again
        buffer.push(pcm_data, seq=0)
        assert buffer.buffer_duration_ms() > 1000  # Full segment + new chunk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])