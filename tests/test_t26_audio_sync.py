"""音频-口型同步验收（/autoplan E-sync 重构）。

性质（按构造）：
1. 音频块在展示时刻推给播放器（帧锁定）——音频位置 ≡ 展示帧位置 ≡ 唇形位置；
2. 推理落后 → HOLD：不分配音频块，音频等待（口型绝不滞后于音频）；
3. 领先量仅剩 DAC 延迟（≤ ~40ms，满足"口型可比音频快 100ms 以内"）；
4. underrun 重发帧不推音频（音画同步停）。
"""
import time

import pytest
from unittest.mock import MagicMock

import numpy as np
import torch

from realtime_serve.output_buffer import FrameBuffer, OutputStreamer
from realtime_serve.render_loop import RenderLoop


def _loop_with_audio(T=10):
    """真 RenderLoop + 真 FrameBuffer + mock 推理；lip 系数一次性填满。"""
    loop = RenderLoop(target_fps=25)
    loop._running = True

    orifra = np.zeros((8, 8, 3), dtype=np.uint8)
    mock_char = MagicMock()
    mock_char.get_frame.return_value = (orifra, MagicMock(), MagicMock(),
                                        MagicMock(), None)
    mock_char.total_frames = 100
    loop.set_character_data(mock_char)

    loop._lip_coeffs = torch.randn(T, 20)
    loop._audio_chunks = [np.full(640, i, dtype=np.float32) for i in range(T)]
    loop._total_lip_frames = T
    loop._lip_filled = T
    loop._audio_arm_start = True

    loop.model_manager = MagicMock()
    loop.model_manager.infer_frame.return_value = torch.randn(1, 3, 256, 256)
    loop.compositor = MagicMock()
    loop.compositor.composite.return_value = orifra
    loop.frame_buffer = FrameBuffer(capacity=16)
    return loop


def test_end_to_end_chunks_follow_display_order():
    """生产 N 帧 → 流线程展示 → 播放器按展示顺序收到 chunk 0..N-1，reset 一次。"""
    T = 6
    loop = _loop_with_audio(T)
    player = MagicMock()
    streamer = OutputStreamer(loop.frame_buffer, target_fps=1000,
                              sender=None, audio_player=player,
                              preroll_frames=0)
    for _ in range(T):
        loop._process_frame()

    streamer.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and player.push_pcm.call_count < T:
            time.sleep(0.01)
        assert player.push_pcm.call_count == T
        pushed = [c.args[0] for c in player.push_pcm.call_args_list]
        assert [float(c[0]) for c in pushed] == list(range(T))  # 展示序 = 唇形序
        player.reset.assert_called_once()  # audio_start 仅首帧
    finally:
        streamer.stop()


def test_hold_audio_waits_for_lips():
    """推理落后 → 无音频块出帧（音频等待）；补齐后从断点继续，不跳不重。"""
    loop = _loop_with_audio(T=10)
    loop._lip_filled = 3  # 仅 3 帧就绪
    player = MagicMock()

    metas = []
    for _ in range(6):
        audio_meta = None
        result = loop._next_lip_frame()
        if result is None:
            loop._get_idle_lip_coeff()
        elif result[0] == 'hold':
            loop._get_idle_lip_coeff()
        else:
            _, alpha, idx, first, chunk = result
            if chunk is not None:
                audio_meta = {'audio': chunk, 'audio_start': bool(first)}
        metas.append(audio_meta)

    # 前 3 帧带 chunk（0,1,2），后 3 帧 HOLD → 无 chunk（音频等待）
    advanced = [m for m in metas if m is not None]
    assert [float(m['audio'][0]) for m in advanced] == [0.0, 1.0, 2.0]
    assert advanced[0]['audio_start'] is True
    assert all(m is None for m in metas[3:])

    # 推理补齐后从 3 继续（不跳帧不重复——音频与唇形保持锁定）
    loop._lip_filled = 10
    r = loop._next_lip_frame()
    assert r[2] == 3 and float(r[4][0]) == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
