"""Unit tests for realtime_serve.output_buffer."""
import time
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from realtime_serve.output_buffer import FrameBuffer, OutputStreamer


def _frame(val=0, shape=(8, 8, 3)):
    return np.full(shape, val, dtype=np.uint8)


class TestFrameBuffer:
    def test_put_get_fifo_order(self):
        fb = FrameBuffer(capacity=4)
        for i in range(3):
            assert fb.put(_frame(i), meta={'i': i}) is True
        for i in range(3):
            frame, meta = fb.get()
            assert meta['i'] == i
        assert fb.get(timeout=0.01) is None

    def test_capacity_blocks_put(self):
        fb = FrameBuffer(capacity=2)
        fb.put(_frame(1))
        fb.put(_frame(2))

        result = []
        t = threading.Thread(target=lambda: result.append(fb.put(_frame(3), timeout=0.05)))
        t.start()
        t.join(timeout=0.2)
        assert result == [False]

    def test_get_blocks_until_item(self):
        fb = FrameBuffer(capacity=2)
        result = []

        def producer():
            time.sleep(0.02)
            fb.put(_frame(42), meta={'k': 42})

        def consumer():
            item = fb.get(timeout=0.2)
            result.append(item)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        frame, meta = result[0]
        assert meta['k'] == 42

    def test_flush_keeps_reserve(self):
        fb = FrameBuffer(capacity=8, reserve=3)
        for i in range(8):
            fb.put(_frame(i), meta={'i': i})
        dropped = fb.flush()
        assert dropped == 5
        assert fb.depth == 3
        # Remaining frames are the newest three.
        for i in range(5, 8):
            frame, meta = fb.get()
            assert meta['i'] == i

    def test_close_wakes_blocked_put(self):
        fb = FrameBuffer(capacity=1)
        fb.put(_frame(1))
        result = []

        def blocked_put():
            result.append(fb.put(_frame(2), timeout=1.0))

        t = threading.Thread(target=blocked_put)
        t.start()
        time.sleep(0.02)
        fb.close()
        t.join(timeout=0.2)
        assert result == [False]

    def test_close_wakes_blocked_get(self):
        fb = FrameBuffer(capacity=2)
        result = []

        def blocked_get():
            result.append(fb.get(timeout=1.0))

        t = threading.Thread(target=blocked_get)
        t.start()
        time.sleep(0.02)
        fb.close()
        t.join(timeout=0.2)
        assert result == [None]

    def test_close_releases_frames(self):
        fb = FrameBuffer(capacity=4)
        for i in range(4):
            fb.put(_frame(i))
        fb.close()
        assert fb.depth == 0
        assert fb.is_closed

    def test_capacity_validation(self):
        with pytest.raises(ValueError):
            FrameBuffer(capacity=0)
        with pytest.raises(ValueError):
            FrameBuffer(capacity=301)


class FakeSender:
    def __init__(self):
        self.frames = []
        self.fail_next = False

    def __call__(self, frame):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("boom")
        self.frames.append(frame.copy())
        return True


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestOutputStreamer:
    def test_preroll_then_output(self):
        fb = FrameBuffer(capacity=8)
        sender = FakeSender()
        streamer = OutputStreamer(fb, target_fps=100, sender=sender, preroll_frames=2)
        streamer.start()
        try:
            time.sleep(0.03)  # should still be prerolling
            assert streamer.is_prerolling is True

            fb.put(_frame(1))
            fb.put(_frame(2))
            time.sleep(0.15)
            assert streamer.is_prerolling is False
            assert len(sender.frames) >= 2
        finally:
            streamer.stop()

    def test_underrun_resends_last_frame(self):
        fb = FrameBuffer(capacity=4)
        sender = FakeSender()
        streamer = OutputStreamer(fb, target_fps=1000, sender=sender, preroll_frames=0)
        streamer.start()
        try:
            fb.put(_frame(1))
            time.sleep(0.01)
            # No more frames: underrun should resend frame 1.
            time.sleep(0.02)
            assert streamer.underruns >= 1
            assert len(sender.frames) >= 2
            # The last sent frame should still be frame 1.
            assert sender.frames[-1][0, 0, 0] == 1
        finally:
            streamer.stop()

    def test_no_camera_drops_but_counts(self):
        fb = FrameBuffer(capacity=4)
        streamer = OutputStreamer(fb, target_fps=1000, sender=None, preroll_frames=0)
        streamer.start()
        try:
            fb.put(_frame(7))
            time.sleep(0.02)
            assert streamer.output_frames >= 1
        finally:
            streamer.stop()

    def test_audio_chunk_pushed_at_display(self):
        """帧锁定音频（E-sync）：展示帧携带的 640 采样块在展示时刻推给播放器；
        audio_start 先 reset 再推块。"""
        fb = FrameBuffer(capacity=4)
        sender = FakeSender()
        player = MagicMock()

        streamer = OutputStreamer(fb, target_fps=1000, sender=sender,
                                  audio_player=player, preroll_frames=0)
        streamer.start()
        try:
            chunk0 = np.zeros(640, dtype=np.float32)
            chunk1 = np.ones(640, dtype=np.float32)
            fb.put(_frame(1), meta={'audio': chunk0, 'audio_start': True})
            fb.put(_frame(2), meta={'audio': chunk1})
            time.sleep(0.03)
            player.reset.assert_called_once()
            assert player.push_pcm.call_count == 2
            pushed = [c.args[0] for c in player.push_pcm.call_args_list]
            assert np.array_equal(pushed[0], chunk0)
            assert np.array_equal(pushed[1], chunk1)
        finally:
            streamer.stop()

    def test_underrun_resend_pushes_no_audio(self):
        """underrun 重发帧无 meta → 不推音频（音频与画面同步停）。"""
        fb = FrameBuffer(capacity=4)
        player = MagicMock()
        streamer = OutputStreamer(fb, target_fps=1000, sender=None,
                                  audio_player=player, preroll_frames=0)
        streamer.start()
        try:
            chunk0 = np.zeros(640, dtype=np.float32)
            fb.put(_frame(1), meta={'audio': chunk0})
            time.sleep(0.05)  # 缓冲空 → underrun 重发（无 meta）
            assert player.push_pcm.call_count == 1  # 只有首块
            assert streamer.underruns >= 1
        finally:
            streamer.stop()

    def test_sender_exception_isolated(self):
        fb = FrameBuffer(capacity=4)
        sender = FakeSender()
        sender.fail_next = True
        streamer = OutputStreamer(fb, target_fps=1000, sender=sender, preroll_frames=0)
        streamer.start()
        try:
            fb.put(_frame(1))
            time.sleep(0.02)
            assert streamer.output_frames >= 1
        finally:
            streamer.stop()
