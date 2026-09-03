"""RealtimeService facade 测试（/autoplan T2：生命周期次序 M12 + 方法面 M1/M23）。

assemble=False + components 注入（M5）——不加载真实模型/角色数据。
"""
import math
from unittest.mock import MagicMock

import pytest

from realtime_serve.service import RealtimeService


class FakePack:
    def __init__(self, segments=("1", "happy"), current="1"):
        self.segments = list(segments)
        self.current_segment = current
        self.total_frames = 5
        self.resolution = (1920, 1080)
        self.prefetch_calls = []
        self.eyes_enabled = None

    def prefetch_segment(self, seg):
        self.prefetch_calls.append(seg)

    def pop_prefetch(self, seg):
        return None

    @property
    def prefetch_status(self):
        return {"state": "idle", "segment": None}

    def set_eyes_enabled(self, enabled):
        self.eyes_enabled = enabled

    def switch_segment(self, seg, prefetched=None):
        self.current_segment = seg
        return True


class FakeGaze:
    def __init__(self):
        self._convergence_h = 0.05
        self.smooth_alpha = 0.4
        self._graph_ready = False
        self.corrected_frames = 0
        self.blink_skips = 0
        self.nan_skips = 0
        self.error_skips = 0

    @property
    def convergence_h(self):
        return self._convergence_h


def _components(pack=None, gaze=None, face_repair=None):
    rl = MagicMock()
    rl.actual_fps = 24.9
    rl.frame_index = 0
    rl.gaze_mode = "on"
    rl.face_repair_mode = "on"
    rl._scheduler = None
    rl.pending_switch = None
    os_ = MagicMock()
    os_.stats.return_value = {"target_fps": 25, "underruns": 0,
                              "preroll_state": "complete"}
    fb = MagicMock()
    fb.stats.return_value = {"depth": 8, "capacity": 16}
    ab = MagicMock()
    ab.buffer_duration_ms.return_value = 120.0
    uc = MagicMock()
    uc.width, uc.height = 1080, 1920
    return {
        "model_manager": None,
        "render_loop": rl,
        "output_streamer": os_,
        "frame_buffer": fb,
        "unity_capture": uc,
        "audio_buffer": ab,
        "audio_processor": None,
        "compositor": None,
        "face_repair": face_repair,
        "audio_player": MagicMock(),
        "character_data": pack if pack is not None else FakePack(),
        "gaze_corrector": gaze,
    }


def _service(**kw) -> RealtimeService:
    return RealtimeService(None, components=_components(**kw), assemble=False)


class TestLifecycle:
    def test_start_order_streamer_before_render_loop(self):
        # M12：streamer.start 必须先于 render_loop.start（统一 attach 到一个
        # MagicMock 才能比较调用顺序）
        import unittest.mock as m
        svc = _service()
        mgr = m.MagicMock()
        mgr.attach_mock(svc.output_streamer, "os")
        mgr.attach_mock(svc.render_loop, "rl")
        svc.start()
        names = [c[0] for c in mgr.mock_calls]
        assert names.index("os.start") < names.index("rl.start")

    def test_double_start_noop(self):
        svc = _service()
        svc.start()
        svc.start()
        assert svc.output_streamer.start.call_count == 1
        assert svc.render_loop.start.call_count == 1

    def test_stop_order_and_idempotent(self):
        svc = _service()
        svc.start()
        svc.stop()
        svc.stop()
        import unittest.mock as m
        assert svc.render_loop.stop.call_count == 1
        assert svc.output_streamer.stop.call_count == 1
        assert svc.audio_player.close.call_count == 1
        assert svc.unity_capture.close.call_count == 1
        assert svc.running is False

    def test_start_after_stop_raises(self):
        svc = _service()
        svc.start()
        svc.stop()
        with pytest.raises(RuntimeError, match="停止"):
            svc.start()


class TestSwitchSegment:
    def test_immediate_delegates_to_render_loop(self):
        svc = _service()
        svc.start()
        # 切换事务快照由 render_loop 在锁内构建（E3）——mock 返回之
        svc.render_loop.switch_segment.return_value = {
            "segment": "happy", "frame_count": 5, "resolution": [1920, 1080]}
        resp = svc.switch_segment("happy")
        kwargs = svc.render_loop.switch_segment.call_args.kwargs
        assert kwargs["source"] == "manual"
        assert kwargs["seamless"] is False
        assert resp["ok"] is True and resp["queued"] is False
        assert resp["frame_count"] == 5
        assert resp["resolution"] == [1920, 1080]

    def test_invalid_at_rejected(self):
        svc = _service()
        with pytest.raises(ValueError, match="at 必须为"):
            svc.switch_segment("happy", at="whenever")

    def test_unknown_segment_rejected(self):
        svc = _service()
        with pytest.raises(ValueError, match="not found"):
            svc.switch_segment("nope")

    def test_segment_end_queues_and_prefetches(self):
        pack = FakePack()
        svc = _service(pack=pack)
        resp = svc.switch_segment("happy", at="segment_end")
        assert pack.prefetch_calls == ["happy"]  # 请求时校验+尽力预取（M16/M15）
        svc.render_loop.queue_segment_switch.assert_called_once_with("happy")
        svc.render_loop.switch_segment.assert_not_called()  # 边界才切
        assert resp == {"ok": True, "at": "segment_end",
                        "pending_segment": "happy",
                        "current_segment": "1"}


class TestGazeParams:
    def test_no_corrector_rejected(self):
        svc = _service()  # gaze_corrector None
        with pytest.raises(RuntimeError, match="gaze_capable=false"):
            svc.set_gaze_params(convergence=0.1)

    def test_valid_convergence_queued(self):
        svc = _service(gaze=FakeGaze())
        resp = svc.set_gaze_params(convergence=0.08)
        svc.render_loop.queue_gaze_params.assert_called_once_with(
            convergence=0.08, smooth_alpha=None, mode=None)
        assert resp == {"ok": True, "effective": "next_frame"}

    def test_nan_and_out_of_range_rejected(self):
        svc = _service(gaze=FakeGaze())
        with pytest.raises(ValueError, match="有限数值"):
            svc.set_gaze_params(convergence=float("nan"))
        with pytest.raises(ValueError, match="有限数值"):
            svc.set_gaze_params(convergence=float("inf"))
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            svc.set_gaze_params(convergence=1.5)

    def test_bad_mode_rejected(self):
        svc = _service(gaze=FakeGaze())
        with pytest.raises(ValueError, match="mode 必须为"):
            svc.set_gaze_params(mode="always")

    def test_empty_call_rejected(self):
        svc = _service(gaze=FakeGaze())
        with pytest.raises(ValueError, match="至少提供"):
            svc.set_gaze_params()


class TestRepairParams:
    def test_bad_mode_rejected(self):
        svc = _service()
        with pytest.raises(ValueError, match="mode 必须为"):
            svc.set_repair_params("sometimes")

    def test_on_without_model_rejected(self):
        svc = _service()  # face_repair None
        with pytest.raises(RuntimeError, match="repair_capable=false"):
            svc.set_repair_params("on")

    def test_off_always_allowed(self):
        svc = _service()  # face_repair None 也可切 off（门控关闭）
        resp = svc.set_repair_params("off")
        assert resp == {"ok": True, "effective": "next_frame"}
        assert svc.render_loop.face_repair_mode == "off"


class TestSetSchedule:
    def test_empty_rejected(self):
        svc = _service()
        with pytest.raises(ValueError, match="非空"):
            svc.set_schedule([])

    def test_unknown_segment_rejected_with_available(self):
        svc = _service()
        with pytest.raises(ValueError, match="可用段"):
            svc.set_schedule(["1", "nope"])

    def test_install_without_align(self):
        pack = FakePack(current="1")
        svc = _service(pack=pack)
        resp = svc.set_schedule(["happy", "1"])
        svc.render_loop.set_segment_sequence.assert_called_once_with(
            ["happy", "1"], mode="list", align=False)
        assert resp["switched"] is False

    def test_align_triggers_switch_report(self):
        pack = FakePack(current="1")
        svc = _service(pack=pack)

        def fake_install(sequence, mode="list", align=True):
            if align:
                pack.current_segment = sequence[0]

        svc.render_loop.set_segment_sequence.side_effect = fake_install
        resp = svc.set_schedule(["happy", "1"], align=True)
        assert resp["switched"] is True
        assert resp["current_segment"] == "happy"


class TestGetStatus:
    def test_shape_with_gaze(self):
        svc = _service(gaze=FakeGaze())
        svc.start()
        status = svc.get_status()
        for key in ("state", "production_fps", "output_fps", "underruns",
                    "preroll_state", "current_segment", "resolution",
                    "available_segments", "gaze_capable", "repair_mode",
                    "repair_capable", "gaze_params", "camera_resolution",
                    "prefetch"):
            assert key in status, key
        assert status["gaze_capable"] is True
        assert status["repair_capable"] is False
        assert status["gaze_params"]["convergence"] == 0.05
        assert status["camera_resolution"] == [1080, 1920]

    def test_shape_without_gaze(self):
        svc = _service()
        status = svc.get_status()
        assert status["gaze_capable"] is False
        assert status["gaze_state"] == "off"
        assert "gaze_params" not in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
