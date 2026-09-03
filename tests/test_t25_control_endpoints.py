"""控制面端点测试（/autoplan T3：/gaze/params + /repair/params + 鉴权 E4）。"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from realtime_serve.api.app import create_app
from realtime_serve.service import RealtimeService


class FakeGaze:
    """与 test_t20_service_lifecycle.py 同款（仓库测试约定：自包含）。"""

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


def _components(gaze=None, face_repair=None, pack=None):
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
    return {
        "model_manager": None,
        "render_loop": rl,
        "output_streamer": os_,
        "frame_buffer": fb,
        "unity_capture": MagicMock(),
        "audio_buffer": ab,
        "audio_processor": None,
        "compositor": None,
        "face_repair": face_repair,
        "audio_player": MagicMock(),
        "character_data": pack if pack is not None else FakePack(),
        "gaze_corrector": gaze,
    }


def _service(gaze=None, face_repair=None, pack=None) -> RealtimeService:
    return RealtimeService(None, components=_components(
        gaze=gaze, face_repair=face_repair, pack=pack), assemble=False)


class TestGazeParamsEndpoint:
    def test_200_when_capable(self):
        svc = _service(gaze=FakeGaze())
        client = TestClient(create_app(svc))
        resp = client.post("/gaze/params", json={"convergence": 0.08})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "effective": "next_frame"}
        svc.render_loop.queue_gaze_params.assert_called_once_with(
            convergence=0.08, smooth_alpha=None, mode=None)

    def test_409_when_not_constructed(self):
        # 启动 off → corrector 未构造（M2/DX-10）
        client = TestClient(create_app(_service()))
        resp = client.post("/gaze/params", json={"convergence": 0.08})
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["problem"] == "功能未启用"
        assert "gaze_capable" in detail["fix"]

    def test_422_schema_violations(self):
        client = TestClient(create_app(_service(gaze=FakeGaze())))
        assert client.post("/gaze/params",
                           json={"convergence": 1.5}).status_code == 422
        assert client.post("/gaze/params",
                           json={"convergence": "NaN"}).status_code == 422
        assert client.post("/gaze/params",
                           json={"mode": "always"}).status_code == 422

    def test_mode_toggle_passes_through(self):
        svc = _service(gaze=FakeGaze())
        client = TestClient(create_app(svc))
        resp = client.post("/gaze/params", json={"mode": "off"})
        assert resp.status_code == 200
        svc.render_loop.queue_gaze_params.assert_called_once_with(
            convergence=None, smooth_alpha=None, mode="off")


class TestRepairParamsEndpoint:
    def test_200_when_capable(self):
        svc = _service(face_repair=MagicMock())
        client = TestClient(create_app(svc))
        resp = client.post("/repair/params", json={"mode": "adaptive"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "effective": "next_frame"}
        assert svc.render_loop.face_repair_mode == "adaptive"

    def test_409_when_not_constructed(self):
        client = TestClient(create_app(_service()))
        resp = client.post("/repair/params", json={"mode": "on"})
        assert resp.status_code == 409

    def test_422_bad_mode(self):
        client = TestClient(create_app(_service()))
        resp = client.post("/repair/params", json={"mode": "sometimes"})
        assert resp.status_code == 422


class TestAuthToken:
    """E4：--api-token 时全路由要求 Bearer；未配置时不要求。"""

    def test_401_without_token(self):
        client = TestClient(create_app(_service(gaze=FakeGaze()),
                                        api_token="secret"))
        assert client.get("/status").status_code == 401
        assert client.post("/gaze/params",
                           json={"mode": "on"}).status_code == 401

    def test_401_wrong_token(self):
        client = TestClient(create_app(_service(gaze=FakeGaze()),
                                        api_token="secret"))
        resp = client.get("/status",
                          headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_200_with_token(self):
        client = TestClient(create_app(_service(gaze=FakeGaze()),
                                        api_token="secret"))
        resp = client.get("/status",
                          headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200

    def test_no_token_configured_open_access(self):
        client = TestClient(create_app(_service()))
        assert client.get("/status").status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
