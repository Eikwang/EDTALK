"""Tests for segment switch API routes (E5 迁移：create_app(service))。"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from realtime_serve.api.app import create_app
from realtime_serve.audio_buffer import AudioBuffer
from realtime_serve.output_buffer import FrameBuffer, OutputStreamer
from realtime_serve.render_loop import RenderLoop
from realtime_serve.service import RealtimeService


def _make_service(character_data, render_loop=None) -> RealtimeService:
    frame_buffer = FrameBuffer(capacity=8)
    return RealtimeService(
        None,
        components={
            'model_manager': None,
            'render_loop': render_loop if render_loop is not None else RenderLoop(),
            'output_streamer': OutputStreamer(
                frame_buffer, target_fps=25, sender=None, preroll_frames=0),
            'frame_buffer': frame_buffer,
            'unity_capture': None,
            'audio_buffer': AudioBuffer(),
            'audio_processor': None,
            'compositor': None,
            'face_repair': None,
            'audio_player': None,
            'character_data': character_data,
            'gaze_corrector': None,
        },
        assemble=False)


def _mock_character():
    mock = MagicMock()
    mock.current_segment = "1"
    mock.total_frames = 100
    mock.resolution = (1920, 1080)
    mock.segments = ["1", "happy", "angry"]
    return mock


class TestSegmentRoutes:
    """Tests for segment switch API routes (T15)."""

    def test_segment_switch_success(self):
        """正常情况：POST /segment/switch 传入合法参数返回成功（事务内快照）"""
        character = _mock_character()
        rl = MagicMock()
        rl.pending_switch = None
        rl.switch_segment.return_value = {
            "segment": "happy", "frame_count": 100, "resolution": [1920, 1080]}

        service = _make_service(character, render_loop=rl)
        client = TestClient(create_app(service))

        response = client.post("/segment/switch", json={"segment": "happy"})

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["segment"] == "happy"
        assert data["frame_count"] == 100
        assert data["queued"] is False

    def test_segment_switch_missing_field(self):
        """异常情况：缺少必填字段时返回 422"""
        service = _make_service(_mock_character())
        client = TestClient(create_app(service))
        response = client.post("/segment/switch", json={})
        assert response.status_code == 422

    def test_segment_switch_empty_body(self):
        """异常情况：空请求体时返回 422"""
        service = _make_service(_mock_character())
        client = TestClient(create_app(service))
        response = client.post("/segment/switch")
        assert response.status_code == 422

    def test_segment_switch_invalid_segment(self):
        """异常情况：切换到不存在的片段返回 404（三段式包络）"""
        service = _make_service(_mock_character())
        client = TestClient(create_app(service))
        response = client.post("/segment/switch", json={"segment": "nonexistent"})

        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail["problem"] == "请求对象不存在"
        assert "available_segments" not in detail  # 可用段在 cause 里回显
        assert "Available" in detail["cause"]

    def test_segment_switch_at_segment_end(self):
        """正常情况：at=segment_end 入队（M16 语义，返回 pending 而非假装已切）"""
        character = _mock_character()
        rl = MagicMock()
        rl.pending_switch = None
        service = _make_service(character, render_loop=rl)
        client = TestClient(create_app(service))

        response = client.post("/segment/switch",
                               json={"segment": "happy", "at": "segment_end"})

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["at"] == "segment_end"
        assert data["pending_segment"] == "happy"
        assert data["current_segment"] == "1"
        rl.queue_segment_switch.assert_called_once_with("happy")

    def test_segment_switch_bad_at_422(self):
        """异常情况：非法 at 值 → pydantic schema 422"""
        service = _make_service(_mock_character())
        client = TestClient(create_app(service))
        response = client.post("/segment/switch",
                               json={"segment": "happy", "at": "whenever"})
        assert response.status_code == 422

    def test_segment_schedule(self):
        """正常情况：POST /segment/schedule 默认不打断当前播放"""
        character = _mock_character()
        rl = MagicMock()
        rl.pending_switch = None
        service = _make_service(character, render_loop=rl)
        client = TestClient(create_app(service))

        response = client.post("/segment/schedule",
                               json={"sequence": ["happy", "1"]})

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["switched"] is False
        rl.set_segment_sequence.assert_called_once_with(
            ["happy", "1"], mode="list", align=False)

    def test_segment_schedule_unknown_segment_400(self):
        """异常情况：调度含未知段返回 400（含可用段回显）"""
        service = _make_service(_mock_character())
        client = TestClient(create_app(service))
        response = client.post("/segment/schedule",
                               json={"sequence": ["1", "nope"]})
        assert response.status_code == 400
        assert "nope" in response.json()["detail"]["cause"]

    def test_status_endpoint(self):
        """正常情况：GET /status 返回服务状态"""
        character = _mock_character()
        service = _make_service(character)
        client = TestClient(create_app(service))
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "fps" in data
        assert "current_segment" in data
        assert "available_segments" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
