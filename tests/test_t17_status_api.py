"""Tests for status API endpoint (E5 迁移：create_app(service))。"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from realtime_serve.api.app import create_app
from realtime_serve.audio_buffer import AudioBuffer
from realtime_serve.output_buffer import FrameBuffer, OutputStreamer
from realtime_serve.render_loop import RenderLoop
from realtime_serve.service import RealtimeService


def _make_service(character_data=None) -> RealtimeService:
    frame_buffer = FrameBuffer(capacity=8)
    return RealtimeService(
        None,
        components={
            'model_manager': None,
            'render_loop': RenderLoop(),
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
    mock.resolution = (1920, 1080)
    mock.segments = ["1", "happy"]
    return mock


class TestStatusEndpoint:
    """Tests for status query API (T17)。/status 为 facade 单一构建器的直通。"""

    def setup_method(self):
        self.service = _make_service(_mock_character())
        self.client = TestClient(create_app(self.service))

    def test_status_returns_all_fields(self):
        """正常情况：GET /status 返回所有必需字段（v0 新字段为增量）"""
        response = self.client.get("/status")
        assert response.status_code == 200
        data = response.json()

        for key in ("state", "fps", "production_fps", "output_fps",
                    "buffer_depth", "buffer_ms", "underruns", "preroll_state",
                    "current_segment", "frame_index", "resolution",
                    "available_segments",
                    # v0 增量字段（只增不改契约）
                    "gaze_capable", "repair_mode", "repair_capable"):
            assert key in data, key

    def test_status_initial_state(self):
        """边界情况：未 start 时 state 合法且 fps 非负"""
        response = self.client.get("/status")
        data = response.json()
        assert data["state"] is not None
        assert data["fps"] >= 0

    def test_status_after_audio_push(self):
        """正常情况：推送音频后 buffer_ms 更新"""
        import base64
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        self.client.post("/audio/push", json={
            "pcm_base64": base64.b64encode(pcm_data).decode('utf-8'),
            "timestamp_ms": 12345})

        data = self.client.get("/status").json()
        assert data["buffer_ms"] > 0

    def test_status_unloaded_character(self):
        """异常情况：角色数据未加载时仍返回 200（字段缺省）"""
        service = _make_service(None)
        client = TestClient(create_app(service))
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["current_segment"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
