"""Gaze/repair 参数加固测试（/autoplan E2：M17 校验 + off 门控守卫 + M23）。

不加载真实 ckpt：GazeCorrector 用 __new__ 骨架只装 set_convergence 触及的
最小状态——校验逻辑与 graph 同步是纯 Python/torch 张量操作，无需模型。
"""
import pytest
import torch

from realtime_serve.gaze_correction import CONVERGENCE_GHOST_DELTA, GazeCorrector
from realtime_serve.render_loop import RenderLoop


def _make_corrector_skeleton() -> GazeCorrector:
    """Minimal GazeCorrector state for set_convergence (no ckpt, CPU)."""
    gc = GazeCorrector.__new__(GazeCorrector)
    gc.device = torch.device('cpu')
    gc._convergence_h = 0.05
    gc._angles = gc._build_angles(0.05)
    gc._graph_ready = False
    gc._static_angles = None
    gc._prev_gen = torch.zeros(2, 3, 8, 8)
    return gc


class TestSetConvergenceValidation:
    """M17 第一/二层：拒绝式校验（NaN/inf 毒化 graph = DoS 级）。"""

    def test_rejects_nan(self):
        gc = _make_corrector_skeleton()
        with pytest.raises(ValueError, match="有限数值"):
            gc.set_convergence(float('nan'))

    def test_rejects_inf(self):
        gc = _make_corrector_skeleton()
        with pytest.raises(ValueError, match="有限数值"):
            gc.set_convergence(float('inf'))

    def test_rejects_out_of_range(self):
        gc = _make_corrector_skeleton()
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            gc.set_convergence(1.5)
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            gc.set_convergence(-1.0001)

    def test_reject_does_not_mutate_state(self):
        gc = _make_corrector_skeleton()
        before = (gc._convergence_h, gc._angles.clone())
        with pytest.raises(ValueError):
            gc.set_convergence(float('nan'))
        assert gc._convergence_h == before[0]
        assert torch.equal(gc._angles, before[1])

    def test_valid_update(self):
        gc = _make_corrector_skeleton()
        gc.set_convergence(0.1)
        assert gc._convergence_h == 0.1
        assert torch.equal(gc._angles, torch.tensor(
            [[-0.1, 0.0, -1.0], [0.1, 0.0, 1.0]]))

    def test_graph_ready_copies_static_angles(self):
        gc = _make_corrector_skeleton()
        gc._graph_ready = True
        gc._static_angles = torch.zeros(2, 3)
        gc.set_convergence(-0.2)
        assert torch.equal(gc._static_angles, gc._angles)


class TestConvergenceGhostClear:
    """M17 第三层：大角度变化丢弃时间平滑历史，避免旧角度残影拖 2-3 帧。"""

    def test_large_change_clears_prev_gen(self):
        gc = _make_corrector_skeleton()
        gc.set_convergence(0.05 + CONVERGENCE_GHOST_DELTA + 0.01)
        assert gc._prev_gen is None

    def test_small_change_keeps_prev_gen(self):
        gc = _make_corrector_skeleton()
        gc.set_convergence(0.06)
        assert gc._prev_gen is not None


class TestOffGateGuards:
    """M2/M23 顺带修复：'off' 门控早退（此前 'off' 与 'on' 同路径放行）。"""

    def _loop(self, **attrs) -> RenderLoop:
        loop = RenderLoop(target_fps=25)
        for k, v in attrs.items():
            setattr(loop, k, v)
        return loop

    def test_repair_off_guard(self):
        # 模型在场（热切场景）但模式 off → 不修复
        loop = self._loop(face_repair_mode='off')
        assert loop._repair_allowed() is False

    def test_repair_on_and_adaptive(self):
        assert self._loop(face_repair_mode='on')._repair_allowed() is True
        # adaptive 无缓冲信息时放行（保持既有行为）
        assert self._loop(face_repair_mode='adaptive')._repair_allowed() is True

    def test_gaze_off_guard(self):
        loop = self._loop(gaze_mode='off')
        assert loop._gaze_allowed() is False

    def test_gaze_on_and_adaptive(self):
        assert self._loop(gaze_mode='on')._gaze_allowed() is True
        assert self._loop(gaze_mode='adaptive')._gaze_allowed() is True

    def test_adaptive_depth_gate(self):
        loop = self._loop(gaze_mode='adaptive')
        loop.frame_buffer = None  # 无缓冲 → True（守卫在深度检查前）
        assert loop._gaze_allowed() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
