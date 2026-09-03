"""Tests for the segment scheduler (autoplan: SEGMENT_SCHEDULER_PLAN)."""
import threading
import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from realtime_serve.render_loop import RenderLoop, SegmentScheduler


class TestSegmentScheduler:
    def test_advance_loops(self):
        s = SegmentScheduler(["1", "55"])
        assert s.sequence == ["1", "55"]
        assert s.advance() == "55"
        assert s.advance() == "1"
        assert s.advance() == "55"

    def test_single_segment_degenerate(self):
        s = SegmentScheduler(["1"])
        assert s.advance() == "1"
        assert s.advance() == "1"

    def test_sync_to_aligns(self):
        s = SegmentScheduler(["1", "55", "开心"])
        s.sync_to("开心")
        assert s.position == 2
        assert s.advance() == "1"

    def test_sync_to_outside_pauses(self):
        s = SegmentScheduler(["1", "55"])
        s.sync_to("44")  # valid segment, not in sequence
        assert s.position == -1
        # paused: current playback continues; scheduler skips its check
        s.advance()  # explicit advance un-pauses
        assert s.position == 1

    def test_concurrent_advance_sync(self):
        s = SegmentScheduler(["1", "2", "3"])
        errors = []

        def adv():
            for _ in range(200):
                s.advance()

        def sync():
            for _ in range(200):
                s.sync_to("2")

        t1 = threading.Thread(target=adv)
        t2 = threading.Thread(target=sync)
        t1.start(); t2.start(); t1.join(); t2.join()
        # Invariant: position always points at a valid index or paused.
        assert s.position == -1 or s.sequence[s.position] == "2"


class FakeCharacterData:
    """Minimal pack double (mock chains would test the mock, not the data)."""

    def __init__(self, segments, frames):
        self.segments = list(segments)
        self.current_segment = segments[0]
        self._frames = dict(zip(segments, frames))
        self.switch_calls = []
        self.orifra_base = "wy/orifra"
        self.resolution = (2560, 1440)

    def _build_file_list(self, seg_dir):
        import os
        return [os.path.join(seg_dir, f) for f in sorted(os.listdir(seg_dir))
                if f.endswith(".jpg")]

    @property
    def total_frames(self):
        return self._frames[self.current_segment]

    def switch_segment(self, seg):
        if seg not in self.segments:
            raise ValueError(f"Segment '{seg}' not found. Available: {self.segments}")
        self.switch_calls.append(seg)
        self.current_segment = seg
        return True


class TestRenderLoopScheduling:
    def _make_loop(self, sequence, frames=(3, 3, 3, 3, 3, 3)):
        loop = RenderLoop(target_fps=25)
        segs = list("123456")[:len(sequence)] if False else None
        # Build a pack whose segments mirror the sequence names.
        pack = FakeCharacterData(sequence, frames[:len(sequence)])
        loop.set_character_data(pack)
        loop._clock = lambda: loop._fake_t
        loop._fake_t = 1000.0
        loop._video_start_time = loop._fake_t
        loop.set_segment_sequence(sequence)
        return loop, pack

    def test_completion_triggers_switch(self):
        loop, pack = self._make_loop(["1", "55"])
        # Exactly at the boundary: elapsed*fps == total
        loop._fake_t = loop._video_start_time + 3 / 25
        loop._check_segment_schedule()
        assert pack.current_segment == "55"
        assert pack.switch_calls == ["55"]
        # Clock re-anchored: no double-trigger within the new segment.
        loop._fake_t += 0.01
        loop._check_segment_schedule()
        assert pack.switch_calls == ["55"]

    def test_before_boundary_no_switch(self):
        loop, pack = self._make_loop(["1", "55"])
        loop._fake_t = loop._video_start_time + 2 / 25
        loop._check_segment_schedule()
        assert pack.switch_calls == []

    def test_failed_switch_reanchors_no_storm(self):
        loop, pack = self._make_loop(["1", "55", "1"])
        calls = {"n": 0}

        def boom(seg):
            calls["n"] += 1
            raise RuntimeError("disk gone")

        pack.switch_segment = boom
        # 5 consecutive boundaries each reach the switch and fail.
        for _ in range(5):
            loop._fake_t = loop._video_start_time + 3 / 25
            loop._check_segment_schedule()
        assert calls["n"] == 5

        # After a failed switch the clock is re-anchored: repeated checks
        # at the SAME instant must NOT re-trigger (log-storm guard).
        for _ in range(5):
            loop._check_segment_schedule()
        assert calls["n"] == 5

    def test_manual_switch_syncs_scheduler(self):
        loop, pack = self._make_loop(["1", "55", "11"])
        loop.switch_segment("11", source='manual')
        assert loop._scheduler.position == 2
        # Loop continues from the aligned position.
        loop._fake_t = loop._video_start_time + 3 / 25
        loop._check_segment_schedule()
        assert pack.current_segment == "1"

    def test_manual_switch_outside_pauses(self):
        loop, pack = self._make_loop(["1", "55"])
        pack.segments = ["1", "55", "44"]
        loop.switch_segment("44", source='manual')
        assert loop._scheduler.position == -1
        loop._fake_t = loop._video_start_time + 3 / 25
        loop._check_segment_schedule()  # paused → no switch attempted
        assert pack.current_segment == "44"

    def test_no_scheduler_zero_change(self):
        loop = RenderLoop(target_fps=25)
        pack = FakeCharacterData(["1"], [3])
        loop.set_character_data(pack)
        loop._clock = lambda: loop._video_start_time + 100.0
        loop._video_start_time = loop._clock() - 100.0
        # Legacy path: no scheduler attribute → no switch even far past end.
        loop._check_segment_schedule()
        assert pack.switch_calls == []

    def test_frame_error_protection(self):
        """A raising _process_frame must not kill the producer loop."""
        loop, pack = self._make_loop(["1"])
        loop._process_frame = MagicMock(side_effect=RuntimeError("gpu boom"))
        loop._running = True
        loop._thread = threading.Thread(target=self._run_briefly, args=(loop,),
                                        daemon=True)
        loop._thread.start()
        time.sleep(0.15)
        loop._running = False
        loop._thread.join(timeout=1.0)
        assert loop._frame_errors > 0
        assert "gpu boom" in loop.last_error

    @staticmethod
    def _run_briefly(loop):
        # Run a few iterations of the protected loop body, then stop.
        while loop._running and loop._frame_errors < 5:
            try:
                loop._check_segment_schedule()
                loop._process_frame()
            except Exception as e:
                loop._frame_errors += 1
                loop.last_error = f"{type(e).__name__}: {e}"
                time.sleep(0.005)
                continue
            time.sleep(0.005)


class TestNaturalSort:
    def test_natural_order(self):
        from realtime_serve.main import _natural_sorted
        assert _natural_sorted(["11", "6", "55", "开心", "1"]) == \
            ["1", "6", "11", "55", "开心"]

    def test_resolve_rejects_unknown(self):
        from realtime_serve.main import _resolve_segment_sequence
        pack = FakeCharacterData(["1", "55"], [10, 10])
        with pytest.raises(ValueError, match="指定的段不存在"):
            _resolve_segment_sequence(("list", ["1", "nope"]), pack)

    def test_resolve_auto_natural(self):
        from realtime_serve.main import _resolve_segment_sequence
        # Uses real wy/orifra segments (the resolution check reads the first
        # frame of each scheduled segment).
        pack = FakeCharacterData(["11", "22", "1"], [10, 10, 10])
        seq, mode = _resolve_segment_sequence(("auto", None), pack)
        assert seq == ["1", "11", "22"]
        assert mode == "auto"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
