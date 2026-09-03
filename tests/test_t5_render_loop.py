"""Tests for RenderLoop module — producer with output buffering."""
import time
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from realtime_serve.render_loop import RenderLoop


def _make_mock_audio_processor(mel_input, bs, T):
    """Create a mock AudioProcessor with fixed pcm_to_mel return values."""
    mock = MagicMock()
    mock.pcm_to_mel.return_value = (mel_input, bs, T)
    return mock


class TestRenderLoop:
    """Tests for RenderLoop class."""

    def test_initialization(self):
        loop = RenderLoop(target_fps=25)
        assert loop.target_fps == 25
        assert loop.frame_interval == 1.0 / 25
        assert loop._running is False
        assert loop.actual_fps == 0.0

    def test_start_stop(self):
        loop = RenderLoop(target_fps=25)
        loop.start()
        assert loop._running is True
        loop.stop()
        assert loop._running is False

    def test_run_loop_exits_on_stop(self):
        loop = RenderLoop(target_fps=25)
        with patch.object(loop, '_process_frame'):
            loop.start()
            time.sleep(0.05)
            loop.stop()
            time.sleep(0.15)
            assert loop._running is False

    def test_get_current_pack(self):
        loop = RenderLoop(target_fps=25)
        assert loop.get_current_pack() is None
        mock_char = MagicMock()
        loop.set_character_data(mock_char)
        assert loop.get_current_pack() is mock_char

    def test_process_full_audio_stores_lip_coeffs_and_pcm(self):
        loop = RenderLoop(target_fps=25)
        loop.frame_buffer = MagicMock()
        loop.frame_buffer.flush.return_value = 0
        loop.output_streamer = MagicMock()

        T = 50
        mel_input = torch.randn(1, T, 1, 80, 16)
        mock_audio_processor = _make_mock_audio_processor(mel_input, 1, T)

        mock_model_manager = MagicMock()
        mock_model_manager.infer_lip.return_value = torch.randn(T, 20)

        loop.audio_processor = mock_audio_processor
        loop.model_manager = mock_model_manager

        pcm = np.random.randn(32000).astype(np.float32)  # T*640 = 32000 精确
        loop.process_full_audio(pcm)

        assert loop._lip_coeffs is not None
        assert loop._total_lip_frames == T
        assert loop._lip_coeffs.shape[0] == T
        # 帧锁定音频：每帧一块 640 采样（16kHz/25fps）
        assert loop._audio_chunks is not None
        assert len(loop._audio_chunks) == T
        assert all(c.shape == (640,) for c in loop._audio_chunks)
        assert loop._audio_arm_start is True
        assert loop._last_rendered_lip_idx == -1
        # Buffer is flushed and preroll is re-armed.
        loop.frame_buffer.flush.assert_called_once()
        loop.output_streamer.reset_preroll.assert_called_once()

    def test_process_full_audio_pads_short_pcm(self):
        """PCM 短于 T*640 → 末块补零对齐（帧数由 mel 决定）。"""
        loop = RenderLoop(target_fps=25)
        loop.frame_buffer = MagicMock()
        loop.frame_buffer.flush.return_value = 0
        loop.output_streamer = MagicMock()

        T = 10
        mel_input = torch.randn(1, T, 1, 80, 16)
        loop.audio_processor = _make_mock_audio_processor(mel_input, 1, T)
        loop.model_manager = MagicMock()
        loop.model_manager.infer_lip.return_value = torch.randn(T, 20)

        loop.process_full_audio(np.random.randn(5000).astype(np.float32))

        assert len(loop._audio_chunks) == T
        assert loop._audio_chunks[-1].shape == (640,)  # 末块补零对齐
        assert loop._audio_chunks[0].shape == (640,)

    def test_frame_locked_lip_consumption(self):
        """单调消费：每次 _next_lip_frame 前进一帧并携带对应音频块。"""
        loop = RenderLoop(target_fps=25)

        T = 5
        loop._lip_coeffs = torch.randn(T, 20)
        loop._audio_chunks = [np.full(640, i, dtype=np.float32) for i in range(T)]
        loop._total_lip_frames = T
        loop._lip_filled = T
        loop._audio_arm_start = True

        seen = []
        for _ in range(T):
            result = loop._next_lip_frame()
            assert result[0] == 'play'
            _, coeff, idx, first, chunk = result
            assert coeff.shape == (1, 20)
            seen.append((idx, first, float(chunk[0])))

        assert [i for i, _, _ in seen] == [0, 1, 2, 3, 4]
        assert seen[0][1] is True and all(f is False for _, f, _ in seen[1:])
        assert [c for _, _, c in seen] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_hold_when_inference_behind(self):
        """推理落后于展示 → HOLD（闭嘴、不分配音频块、计数不推进）——音频等待唇形。"""
        loop = RenderLoop(target_fps=25)
        T = 50
        loop._lip_coeffs = torch.zeros(T, 20)
        loop._audio_chunks = [np.zeros(640, dtype=np.float32) for _ in range(T)]
        loop._total_lip_frames = T
        loop._lip_filled = 2  # 仅前 2 帧就绪
        loop._audio_arm_start = True

        r0 = loop._next_lip_frame()
        r1 = loop._next_lip_frame()
        assert r0[0] == 'play' and r0[2] == 0 and r0[3] is True
        assert r1[0] == 'play' and r1[2] == 1

        for _ in range(5):
            assert loop._next_lip_frame()[0] == 'hold'
        assert loop._last_rendered_lip_idx == 1  # 不推进

        # 推理补上第 3 帧后恢复推进（从 2 开始，不跳不重）
        loop._lip_filled = 3
        r2 = loop._next_lip_frame()
        assert r2[0] == 'play' and r2[2] == 2

    def test_audio_finished_holds_last_coeff_then_idle(self):
        """播完 → 保持末帧口形 ~1s（无音频）→ 回 idle 并清理状态。"""
        loop = RenderLoop(target_fps=25)

        T = 2
        lip_coeffs = torch.randn(T, 20)
        loop._lip_coeffs = lip_coeffs
        loop._audio_chunks = [np.zeros(640, dtype=np.float32) for _ in range(T)]
        loop._total_lip_frames = T
        loop._lip_filled = T

        # 消费完两帧语音（各带音频块）
        r0 = loop._next_lip_frame()
        assert r0[2] == 0 and r0[4] is not None
        r1 = loop._next_lip_frame()
        assert r1[2] == 1 and r1[4] is not None

        # hold 窗口：保持末帧口形、无音频块
        hold_frames = loop._audio_hold_frames
        for i in range(hold_frames):
            result = loop._next_lip_frame()
            assert result[0] == 'play'
            assert result[2] == T - 1
            assert torch.equal(result[1], lip_coeffs[-1:])
            assert result[4] is None

        # hold 窗口结束 → idle，状态清理
        assert loop._next_lip_frame() is None
        assert loop._lip_coeffs is None
        assert loop._audio_chunks is None

    def test_idle_mode_uses_idle_lip_coeff(self):
        loop = RenderLoop(target_fps=25)
        loop._running = True

        orifra = np.zeros((1920, 1080, 3), dtype=np.uint8)
        mock_char = MagicMock()
        mock_char.get_frame.return_value = (orifra, MagicMock(), MagicMock(), MagicMock(), None)
        mock_char.total_frames = 100
        loop.set_character_data(mock_char)
        loop._lip_coeffs = None

        idle_pcm = np.random.randn(3200).astype(np.float32) * 0.004
        mock_audio_processor = MagicMock()
        mock_audio_processor.consume_idle_pcm.return_value = idle_pcm
        mock_audio_processor.pcm_to_mel.return_value = (torch.randn(1, 5, 1, 80, 16), 1, 5)

        mock_model_manager = MagicMock()
        mock_model_manager.infer_lip.return_value = torch.randn(5, 20)
        mock_model_manager.infer_frame.return_value = torch.randn(1, 3, 256, 256)

        mock_compositor = MagicMock()
        mock_compositor.composite.return_value = np.zeros((1920, 1080, 3), dtype=np.uint8)

        mock_frame_buffer = MagicMock()

        loop.audio_processor = mock_audio_processor
        loop.model_manager = mock_model_manager
        loop.compositor = mock_compositor
        loop.frame_buffer = mock_frame_buffer

        loop._process_frame()
        mock_model_manager.infer_lip.assert_called_once()
        mock_model_manager.infer_frame.assert_called_once()
        mock_frame_buffer.put.assert_called_once()

    def test_idle_lip_coeff_cached(self):
        loop = RenderLoop(target_fps=25)
        loop._running = True

        orifra = np.zeros((1920, 1080, 3), dtype=np.uint8)
        mock_char = MagicMock()
        mock_char.get_frame.return_value = (orifra, MagicMock(), MagicMock(), MagicMock(), None)
        mock_char.total_frames = 100
        loop.set_character_data(mock_char)
        loop._lip_coeffs = None

        mock_audio_processor = MagicMock()
        mock_audio_processor.consume_idle_pcm.return_value = np.random.randn(3200).astype(np.float32) * 0.004
        mock_audio_processor.pcm_to_mel.return_value = (torch.randn(1, 5, 1, 80, 16), 1, 5)

        mock_model_manager = MagicMock()
        mock_model_manager.infer_lip.return_value = torch.randn(5, 20)
        mock_model_manager.infer_frame.return_value = torch.randn(1, 3, 256, 256)

        mock_compositor = MagicMock()
        mock_compositor.composite.return_value = np.zeros((1920, 1080, 3), dtype=np.uint8)

        loop.audio_processor = mock_audio_processor
        loop.model_manager = mock_model_manager
        loop.compositor = mock_compositor
        loop.frame_buffer = MagicMock()

        for _ in range(5):
            loop._process_frame()

        assert mock_model_manager.infer_lip.call_count == 1
        assert mock_model_manager.infer_frame.call_count == 5

    def test_no_character_data_logs_error(self):
        loop = RenderLoop(target_fps=25)
        loop._running = True
        loop.frame_buffer = MagicMock()
        loop._process_frame()
        loop.frame_buffer.put.assert_not_called()

    def test_render_with_audio_sync(self):
        loop = RenderLoop(target_fps=25)
        loop._running = True

        orifra = np.zeros((1920, 1080, 3), dtype=np.uint8)
        mock_char = MagicMock()
        mock_char.get_frame.return_value = (orifra, MagicMock(), MagicMock(), MagicMock(), None)
        mock_char.total_frames = 100
        loop.set_character_data(mock_char)

        T = 50
        lip_coeffs = torch.randn(T, 20)
        loop._lip_coeffs = lip_coeffs
        loop._audio_chunks = [np.zeros(640, dtype=np.float32) for _ in range(T)]
        loop._total_lip_frames = T
        loop._lip_filled = T

        mock_model_manager = MagicMock()
        mock_model_manager.infer_frame.return_value = torch.randn(1, 3, 256, 256)

        mock_compositor = MagicMock()
        mock_compositor.composite.return_value = np.zeros((1920, 1080, 3), dtype=np.uint8)

        loop.model_manager = mock_model_manager
        loop.compositor = mock_compositor
        loop.frame_buffer = MagicMock()
        loop.frame_buffer.depth = 0
        loop.frame_buffer.capacity = 8

        loop._process_frame()
        mock_model_manager.infer_frame.assert_called_once()
        loop.frame_buffer.put.assert_called_once()

    def test_audio_chunk_rides_first_frame_meta(self):
        """帧锁定：音频块随帧 meta 出帧，audio_start 仅在首个推进帧。"""
        loop = RenderLoop(target_fps=25)
        loop._running = True

        orifra = np.zeros((1920, 1080, 3), dtype=np.uint8)
        mock_char = MagicMock()
        mock_char.get_frame.return_value = (orifra, MagicMock(), MagicMock(), MagicMock(), None)
        mock_char.total_frames = 100
        loop.set_character_data(mock_char)

        T = 5
        loop._lip_coeffs = torch.randn(T, 20)
        loop._audio_chunks = [np.full(640, i, dtype=np.float32) for i in range(T)]
        loop._total_lip_frames = T
        loop._lip_filled = T
        loop._audio_arm_start = True

        mock_model_manager = MagicMock()
        mock_model_manager.infer_frame.return_value = torch.randn(1, 3, 256, 256)
        mock_compositor = MagicMock()
        mock_compositor.composite.return_value = orifra
        loop.model_manager = mock_model_manager
        loop.compositor = mock_compositor
        loop.frame_buffer = MagicMock()
        loop.frame_buffer.depth = 0
        loop.frame_buffer.capacity = 8

        # 第 1 帧：audio_start=True + chunk 0
        loop._process_frame()
        args, kwargs = loop.frame_buffer.put.call_args
        assert kwargs['meta']['audio_start'] is True
        assert float(kwargs['meta']['audio'][0]) == 0.0
        # 第 2 帧：仅 chunk，无 audio_start
        loop._process_frame()
        args, kwargs = loop.frame_buffer.put.call_args
        assert kwargs['meta']['audio_start'] is False
        assert float(kwargs['meta']['audio'][0]) == 1.0

    def test_adaptive_repair_gate(self):
        """Adaptive mode: repair runs only when the buffer has headroom."""
        loop = RenderLoop(target_fps=25)
        loop.face_repair = MagicMock()
        loop.face_repair.repair.side_effect = lambda f: f
        loop.face_repair_mode = 'adaptive'

        loop.frame_buffer = MagicMock()
        loop.frame_buffer.depth = RenderLoop.REPAIR_MIN_BUFFER
        assert loop._repair_allowed() is True

        loop.frame_buffer.depth = RenderLoop.REPAIR_MIN_BUFFER - 1
        assert loop._repair_allowed() is False

        # 'on' mode ignores buffer depth.
        loop.face_repair_mode = 'on'
        loop.frame_buffer.depth = 0
        assert loop._repair_allowed() is True

        # No buffer (mock-legacy path): allow.
        loop.face_repair_mode = 'adaptive'
        loop.frame_buffer = None
        assert loop._repair_allowed() is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
