"""Tests for realtime_serve.gaze_correction (Route C' gaze corrector)."""
import numpy as np
import pytest
import torch

from realtime_serve.gaze_correction import (
    GazeCorrector, EAR_CLOSE, EAR_OPEN, OPEN_CONFIRM_FRAMES, MAX_BLINK_FRAMES,
)

CKPT = "realtime_serve/weights/current_model.ckpt"


def _make_facemark(left_open=True, right_open=True, cx=90.0, cy=110.0):
    """68x2 facemark with controllable eye openness (dlib index order)."""
    fm = np.zeros((68, 2), dtype=np.float32)
    # jaw + nose + mouth points somewhere plausible (not used by gaze logic
    # except stability; fill deterministically).
    fm[:, 0] = np.linspace(40, 200, 68)
    fm[:, 1] = 120.0

    def eye(base_x, open_):
        r = 10.0 if open_ else 1.0
        # dlib order: [outer, top1, top2, inner, bottom2, bottom1]
        pts = np.array([
            [0, 0], [10, r], [20, r], [30, 0], [20, -r], [10, -r]
        ], dtype=np.float32)
        pts[:, 0] += base_x
        pts[:, 1] += cy
        return pts

    fm[36:42] = eye(cx - 30, left_open)
    fm[42:48] = eye(cx + 10, right_open)
    return fm


@pytest.fixture(scope="module")
def corrector_cpu():
    return GazeCorrector(CKPT, device='cpu')


@pytest.fixture(autouse=True)
def _reset_state(corrector_cpu):
    """Blink state machine carries across tests — reset before each."""
    corrector_cpu.reset()
    yield
    corrector_cpu.reset()


class TestEyeState:
    def test_ear_open(self, corrector_cpu):
        fm = _make_facemark(left_open=True, right_open=True)
        assert corrector_cpu._ear(fm[36:42]) > EAR_OPEN

    def test_ear_closed(self, corrector_cpu):
        fm = _make_facemark(left_open=False, right_open=False)
        assert corrector_cpu._ear(fm[36:42]) < EAR_CLOSE

    def test_smooth_small_step(self, corrector_cpu):
        assert corrector_cpu._smooth(100.0, 98.0, 0.4) == pytest.approx(
            0.4 * 100 + 0.6 * 98)
        # Large jump (>20) uses reduced gain 0.3*alpha
        assert corrector_cpu._smooth(140.0, 98.0, 0.4) == pytest.approx(
            0.12 * 140 + 0.88 * 98)

    def test_eye_size_locked(self, corrector_cpu):
        corrector_cpu.reset()
        fm1 = _make_facemark()
        corrector_cpu._eye_state(fm1)
        locked = corrector_cpu._fixed_eye_size
        assert locked is not None and 16 <= locked <= 128

        # Wider eyes next frame must NOT change the locked size.
        fm2 = _make_facemark()
        fm2[36:48, 0] *= 1.5
        corrector_cpu._eye_state(fm2)
        assert corrector_cpu._fixed_eye_size == locked

    def test_blink_state_machine(self, corrector_cpu):
        corrector_cpu.reset()
        # Open frames: correcting.
        l, r = corrector_cpu._eye_state(_make_facemark(True, True))
        assert l is not None
        # Closed frame: gate engages.
        l, r = corrector_cpu._eye_state(_make_facemark(False, False))
        assert l is None and r is None
        # Re-open: needs OPEN_CONFIRM_FRAMES consecutive open frames.
        for _ in range(OPEN_CONFIRM_FRAMES - 1):
            l, r = corrector_cpu._eye_state(_make_facemark(True, True))
            assert l is None  # still gated
        l, r = corrector_cpu._eye_state(_make_facemark(True, True))
        assert l is not None  # confirmed open, back to correcting

    def test_blink_timeout_recovers(self, corrector_cpu):
        corrector_cpu.reset()
        corrector_cpu._eye_state(_make_facemark(False, False))
        # Sustained closure beyond MAX_BLINK_FRAMES must resume correcting.
        for _ in range(MAX_BLINK_FRAMES + 1):
            l, r = corrector_cpu._eye_state(_make_facemark(False, False))
        assert l is not None


class TestCorrect:
    def test_output_invariants(self, corrector_cpu):
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256) * 0.3
        fm = _make_facemark()
        out = corrector_cpu.correct(crop, fm)
        assert out.shape == crop.shape
        assert out.dtype == crop.dtype
        assert out.min() >= -1.5 and out.max() <= 1.5

    def test_no_inplace_mutation(self, corrector_cpu):
        """The preloader buffers are shared: correct() must never mutate input."""
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256) * 0.3
        snapshot = crop.clone()
        corrector_cpu.correct(crop, _make_facemark())
        assert torch.equal(crop, snapshot)

    def test_nan_facemark_bypass(self, corrector_cpu):
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256)
        fm = _make_facemark()
        fm[40, 0] = np.nan
        out = corrector_cpu.correct(crop, fm)
        assert torch.equal(out, crop)
        assert corrector_cpu.nan_skips == 1

    def test_none_facemark_bypass(self, corrector_cpu):
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256)
        out = corrector_cpu.correct(crop, None)
        assert torch.equal(out, crop)

    def test_correction_changes_eye_region(self, corrector_cpu):
        """Sanity: with an open-eye facemark, the eye region actually changes."""
        corrector_cpu.reset()
        g = torch.Generator().manual_seed(42)
        crop = (torch.randn(3, 256, 256, generator=g) * 0.3)
        fm = _make_facemark()
        out = corrector_cpu.correct(crop, fm)
        diff = (out - crop).abs().mean(dim=0)  # per-pixel mean over channels
        eye_mask = torch.zeros(256, 256, dtype=torch.bool)
        pts = fm[36:48]
        x1, y1 = int(pts.min(axis=0)[0]) - 10, int(pts.min(axis=0)[1]) - 10
        x2, y2 = int(pts.max(axis=0)[0]) + 10, int(pts.max(axis=0)[1]) + 10
        eye_mask[max(0, y1):y2, max(0, x1):x2] = True
        eye_delta = diff[eye_mask].mean()
        outside_delta = diff[~eye_mask].mean()
        assert eye_delta > outside_delta * 2
        assert corrector_cpu.corrected_frames == 1

    def test_set_convergence_changes_angles(self, corrector_cpu):
        corrector_cpu.reset()
        old = corrector_cpu._angles.clone()
        corrector_cpu.set_convergence(0.1)
        assert not torch.equal(corrector_cpu._angles, old)
        assert corrector_cpu._convergence_h == 0.1

    def test_reset_clears_state(self, corrector_cpu):
        corrector_cpu.reset()
        corrector_cpu.correct(torch.randn(3, 256, 256), _make_facemark())
        assert corrector_cpu.corrected_frames > 0
        assert corrector_cpu._prev_left is not None
        corrector_cpu.reset()
        assert corrector_cpu.corrected_frames == 0
        assert corrector_cpu._prev_left is None
        assert corrector_cpu._fixed_eye_size is None

    def test_try_capture_graph_on_cpu(self, corrector_cpu):
        assert corrector_cpu.try_capture_graph() is False

    def test_gaze_frame_lookup_table(self, corrector_cpu):
        """查表路径：state 驱动旁路，中心/尺寸来自离线数据。"""
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256) * 0.3
        snapshot = crop.clone()

        # state=1（眨眼）：旁路
        gf_blink = {'state': 1, 'l_center': np.array([90., 110.], dtype=np.float32),
                    'r_center': np.array([130., 110.], dtype=np.float32),
                    'eye_size': np.float32(39.0)}
        out = corrector_cpu.correct(crop, None, gf_blink)
        assert torch.equal(out, crop)
        assert corrector_cpu.blink_skips == 1

        # state=2（无数据）：旁路
        gf_nan = dict(gf_blink, state=2)
        out = corrector_cpu.correct(crop, None, gf_nan)
        assert torch.equal(out, crop)
        assert corrector_cpu.nan_skips == 1

        # state=0（正常）：纠正生效，输入不被修改，计数增长
        gf_ok = dict(gf_blink, state=0)
        out = corrector_cpu.correct(crop, None, gf_ok)
        assert not torch.equal(out, crop)
        assert torch.equal(crop, snapshot)
        assert corrector_cpu.corrected_frames == 1

    def test_gaze_frame_eye_size_used(self, corrector_cpu):
        """查表路径使用离线 eye_size（回归：单眼尺寸而非双眼联合外包络）。"""
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256) * 0.3
        gf_ok = {'state': 0, 'l_center': np.array([90., 110.], dtype=np.float32),
                 'r_center': np.array([130., 110.], dtype=np.float32),
                 'eye_size': np.float32(39.0)}
        out = corrector_cpu.correct(crop, None, gf_ok)
        diff = (out - crop).abs().mean(dim=0)
        # 纠正应集中在眼中心 ±20px 内（39px 框），而非大区域
        y, x = np.mgrid[0:256, 0:256]
        near_l = (np.abs(x - 90) < 25) & (np.abs(y - 110) < 25)
        assert diff[near_l].mean() > diff[~near_l].mean() * 2


class TestRenderLoopGate:
    def test_gaze_allowed_gate(self):
        from realtime_serve.render_loop import RenderLoop
        from unittest.mock import MagicMock
        loop = RenderLoop(target_fps=25)
        loop.gaze_mode = 'adaptive'
        loop.frame_buffer = MagicMock()
        loop.frame_buffer.depth = loop.REPAIR_MIN_BUFFER
        assert loop._gaze_allowed() is True
        loop.frame_buffer.depth = loop.REPAIR_MIN_BUFFER - 1
        assert loop._gaze_allowed() is False
        loop.gaze_mode = 'on'
        assert loop._gaze_allowed() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
