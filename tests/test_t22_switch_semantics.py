"""切换语义测试（/autoplan T4：M16 队列生命周期 + M8 边界不 flush）。

RenderLoop 级验收（EMBEDDING.md 行为语义的可测口径）：
- segment_end：边界处无 flush（缓冲尾帧=合法续播）、无 preroll 重布防；
- immediate：flush 全弃 + preroll 重布防（≤0.3s 量化承诺）；
- pending：legacy（无调度器）也生效；immediate 清除挂起；last-write-wins。
"""
import pytest

from realtime_serve.output_buffer import FrameBuffer
from realtime_serve.render_loop import RenderLoop


class FakeCharacterData:
    """与 test_segment_scheduler.py 同款最小 pack（仓库测试约定：自包含）。"""

    def __init__(self, segments, frames):
        self.segments = list(segments)
        self.current_segment = segments[0]
        self._frames = dict(zip(segments, frames))
        self.switch_calls = []
        self.resolution = (2560, 1440)

    @property
    def total_frames(self):
        return self._frames[self.current_segment]

    def switch_segment(self, seg, prefetched=None):
        if seg not in self.segments:
            raise ValueError(
                f"Segment '{seg}' not found. Available: {self.segments}")
        self.switch_calls.append(seg)
        self.current_segment = seg
        return True


def _loop(sequence, all_segments=None):
    """Build a loop with a fake-clock + real FrameBuffer.

    sequence 非空时安装调度器；all_segments 允许 pending 目标在调度器外
    （legacy 模式验证）。
    """
    loop = RenderLoop(target_fps=25)
    segs = list(all_segments) if all_segments is not None else list(sequence)
    pack = FakeCharacterData(segs, [3] * len(segs))
    loop.set_character_data(pack)
    loop.frame_buffer = FrameBuffer(capacity=8)
    loop._clock = lambda: loop._fake_t
    loop._fake_t = 1000.0
    loop._video_start_time = loop._fake_t
    if sequence:
        loop.set_segment_sequence(list(sequence))
    return loop, pack


def _fill_buffer(loop, n=3):
    for i in range(n):
        assert loop.frame_buffer.put(f"frame{i}")


class TestBoundarySeamlessNoFlush:
    """M8：边界切换（scheduler 推进）不 flush 缓冲；immediate 保持硬切。"""

    def test_auto_advance_keeps_buffer(self):
        loop, pack = _loop(["1", "55"])
        _fill_buffer(loop, 3)
        loop._fake_t = loop._video_start_time + 0.5  # 12.5 帧，越界
        loop._check_segment_schedule()
        assert pack.current_segment == "55"
        assert loop.frame_buffer.depth == 3  # 旧段尾帧保留为合法续播

    def test_immediate_flushes_buffer(self):
        loop, pack = _loop(["1", "55"])
        _fill_buffer(loop, 3)
        loop.switch_segment("55")  # 默认 immediate（非 seamless）
        assert pack.current_segment == "55"
        assert loop.frame_buffer.depth == 0  # E2 硬切：全弃 + preroll 重布防


class TestPendingSegmentEnd:
    """M16：segment_end 队列生命周期。"""

    def test_pending_fires_legacy_no_scheduler(self):
        # legacy：不装调度器（--segments 省略）——pending 仍必须生效（L1）
        loop, pack = _loop([], all_segments=["1", "55"])
        assert loop._scheduler is None
        loop._pending_switch = "55"  # facade 已在请求时校验；直设以测触发
        loop._fake_t = loop._video_start_time + 0.5
        loop._check_segment_schedule()
        assert pack.current_segment == "55"
        assert loop._pending_switch is None

    def test_pending_fires_before_scheduler_and_syncs(self):
        loop, pack = _loop(["1", "55", "22"])
        loop.queue_segment_switch("22")
        assert loop.pending_switch == "22"
        loop._fake_t = loop._video_start_time + 0.5
        loop._check_segment_schedule()
        assert pack.current_segment == "22"
        assert loop._scheduler.position == 2  # 消费即 sync_to（L2 副作用）
        # 后续轮播从 '22' 继续：下一边界 → '1'
        loop._fake_t = loop._video_start_time + 0.5
        loop._check_segment_schedule()
        assert pack.current_segment == "1"

    def test_immediate_cancels_pending(self):
        loop, pack = _loop(["1", "44", "55"])
        loop.queue_segment_switch("55")
        loop.switch_segment("44")  # immediate 取消挂起（L3）
        assert loop.pending_switch is None
        assert pack.current_segment == "44"

    def test_pending_last_write_wins(self):
        loop, pack = _loop(["1", "44", "55"])
        loop.queue_segment_switch("55")
        loop.queue_segment_switch("44")
        assert loop.pending_switch == "44"
        loop._fake_t = loop._video_start_time + 0.5
        loop._check_segment_schedule()
        assert pack.current_segment == "44"

    def test_pending_failed_switch_reanchors(self):
        loop, pack = _loop([], all_segments=["1", "55"])
        calls = {"n": 0}

        def boom(seg, prefetched=None):
            calls["n"] += 1
            raise RuntimeError("disk gone")

        pack.switch_segment = boom
        loop._pending_switch = "55"
        loop._fake_t = loop._video_start_time + 0.5
        loop._check_segment_schedule()
        assert calls["n"] == 1
        assert loop._pending_switch is None  # 已消费，不风暴
        for _ in range(3):
            loop._check_segment_schedule()
        assert calls["n"] == 1  # 重锚后不重复触发


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
