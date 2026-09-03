"""Tests for audio API routes (E5 迁移：create_app(service) 构造注入)。"""
import base64
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from realtime_serve.api.app import create_app
from realtime_serve.audio_buffer import AudioBuffer
from realtime_serve.output_buffer import FrameBuffer, OutputStreamer
from realtime_serve.render_loop import RenderLoop
from realtime_serve.service import RealtimeService


def _make_service(character_data=None, audio_buffer=None,
                  render_loop=None) -> RealtimeService:
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
            'audio_buffer': audio_buffer if audio_buffer is not None else AudioBuffer(),
            'audio_processor': None,
            'compositor': None,
            'face_repair': None,
            'audio_player': None,
            'character_data': character_data,
            'gaze_corrector': None,
        },
        assemble=False)


def _client(service) -> TestClient:
    return TestClient(create_app(service))


class TestAudioRoutes:
    """Tests for audio API routes (T10)."""

    def setup_method(self):
        self.service = _make_service()
        self.client = _client(self.service)

    def test_audio_push_success(self):
        """正常情况：POST /audio/push 传入合法 JSON 返回成功"""
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        pcm_base64 = base64.b64encode(pcm_data).decode('utf-8')

        response = self.client.post("/audio/push", json={
            "pcm_base64": pcm_base64,
            "timestamp_ms": 12345
        })

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "buffer_ms" in data

    def test_audio_push_invalid_base64(self):
        """边界情况：传入非法 base64 字符串时返回 400（三段式包络）"""
        response = self.client.post("/audio/push", json={
            "pcm_base64": "not-valid-base64!!!",
            "timestamp_ms": 12345
        })

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["problem"] == "音频数据解码失败"
        assert "fix" in detail

    def test_audio_push_missing_field(self):
        """异常情况：缺少必填字段时返回 422"""
        response = self.client.post("/audio/push", json={
            "timestamp_ms": 12345
        })
        assert response.status_code == 422

    def test_audio_push_empty_body(self):
        """异常情况：空请求体时返回 422"""
        response = self.client.post("/audio/push")
        assert response.status_code == 422

    def test_audio_push_oversize_413(self, monkeypatch):
        """边界情况：超过音频载荷上限返回 413（E4）"""
        import realtime_serve.api.app as app_module
        monkeypatch.setattr(app_module, "_MAX_AUDIO_BYTES", 100)
        pcm_data = (np.random.rand(3200) * 32767).astype(np.int16).tobytes()
        response = self.client.post("/audio/push", json={
            "pcm_base64": base64.b64encode(pcm_data).decode('utf-8')})
        assert response.status_code == 413

    def test_status_endpoint(self):
        """正常情况：GET /status 返回服务状态"""
        response = self.client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "fps" in data

    def test_audio_push_full_success(self):
        """正常情况：POST /audio/push_full 传入完整音频返回成功"""
        pcm_data = (np.random.rand(16000) * 32767).astype(np.int16).tobytes()
        pcm_base64 = base64.b64encode(pcm_data).decode('utf-8')

        self.service.render_loop.process_full_audio = MagicMock()
        response = self.client.post("/audio/push_full", json={
            "pcm_base64": pcm_base64
        })

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "duration_s" in data
        assert data["duration_s"] >= 0.9
        assert data["processing"] == "async"

    def test_audio_push_full_calls_process_full_audio(self):
        """正常情况：push_full 调用 render_loop.process_full_audio()"""
        mock_fn = MagicMock()
        self.service.render_loop.process_full_audio = mock_fn

        pcm_data = (np.random.rand(16000) * 32767).astype(np.int16).tobytes()
        pcm_base64 = base64.b64encode(pcm_data).decode('utf-8')

        response = self.client.post("/audio/push_full", json={
            "pcm_base64": pcm_base64
        })

        assert response.status_code == 200
        deadline = time.time() + 2.0
        while time.time() < deadline and mock_fn.call_count == 0:
            time.sleep(0.01)
        mock_fn.assert_called_once()
        pcm_arg = mock_fn.call_args[0][0]
        assert pcm_arg.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
