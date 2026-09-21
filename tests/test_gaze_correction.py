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
        crop = (torch.randn(3, 256, 256) * 0.3).clamp(-1, 1)
        fm = _make_facemark()
        out = corrector_cpu.correct(crop, crop, fm)
        assert out.shape == crop.shape
        assert out.dtype == crop.dtype
        assert out.min() >= -1.5 and out.max() <= 1.5

    def test_no_inplace_mutation(self, corrector_cpu):
        """The preloader buffers are shared: correct() must never mutate input."""
        corrector_cpu.reset()
        crop = (torch.randn(3, 256, 256) * 0.3).clamp(-1, 1)
        snapshot = crop.clone()
        corrector_cpu.correct(crop, crop, _make_facemark())
        assert torch.equal(crop, snapshot)

    def test_nan_facemark_bypass(self, corrector_cpu):
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256).clamp(-1, 1)
        fm = _make_facemark()
        fm[40, 0] = np.nan
        out = corrector_cpu.correct(crop, crop, fm)
        assert torch.equal(out, crop)
        assert corrector_cpu.nan_skips == 1

    def test_none_facemark_bypass(self, corrector_cpu):
        corrector_cpu.reset()
        crop = torch.randn(3, 256, 256).clamp(-1, 1)
        out = corrector_cpu.correct(crop, crop, None)
        assert torch.equal(out, crop)

    def test_correction_changes_eye_region(self, corrector_cpu):
        """Sanity: with an open-eye facemark, the eye region actually changes."""
        corrector_cpu.reset()
        g = torch.Generator().manual_seed(42)
        crop = ((torch.randn(3, 256, 256, generator=g) * 0.3).clamp(-1, 1))
        fm = _make_facemark()
        out = corrector_cpu.correct(crop, crop, fm)
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
        corrector_cpu.correct(torch.randn(3, 256, 256).clamp(-1, 1), torch.randn(3, 256, 256).clamp(-1, 1), _make_facemark())
        assert corrector_cpu.corrected_frames > 0
        assert corrector_cpu._prev_left is not None
        corrector_cpu.reset()
        assert corrector_cpu.corrected_frames == 0
        assert corrector_cpu._prev_left is None
        assert corrector_cpu._fixed_eye_size is None

    def test_try_capture_graph_on_cpu(self, corrector_cpu):
        assert corrector_cpu.try_capture_graph() is False

    def test_gaze_frame_lookup_table(self, corrector_cpu):
        """查表路径：state 驱动旁路，中心/尺寸来自离线数据。（legacy
        rendered 模式——bypass 逐位不变；ori 模式的贴回语义见 U7。）"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('rendered')
        crop = (torch.randn(3, 256, 256) * 0.3).clamp(-1, 1)
        snapshot = crop.clone()

        # state=1（眨眼）：旁路
        gf_blink = {'state': 1, 'l_center': np.array([90., 110.], dtype=np.float32),
                    'r_center': np.array([130., 110.], dtype=np.float32),
                    'eye_size': np.float32(39.0)}
        out = corrector_cpu.correct(crop, crop, None, gf_blink)
        assert torch.equal(out, crop)
        assert corrector_cpu.blink_skips == 1

        # state=2（无数据）：旁路
        gf_nan = dict(gf_blink, state=2)
        out = corrector_cpu.correct(crop, crop, None, gf_nan)
        assert torch.equal(out, crop)
        assert corrector_cpu.nan_skips == 1

        # state=0（正常）：纠正生效，输入不被修改，计数增长
        gf_ok = dict(gf_blink, state=0)
        out = corrector_cpu.correct(crop, crop, None, gf_ok)
        assert not torch.equal(out, crop)
        assert torch.equal(crop, snapshot)
        assert corrector_cpu.corrected_frames == 1

    def test_gaze_frame_eye_size_used(self, corrector_cpu):
        """查表路径使用离线 eye_size（回归：单眼尺寸而非双眼联合外包络）。"""
        corrector_cpu.reset()
        crop = (torch.randn(3, 256, 256) * 0.3).clamp(-1, 1)
        gf_ok = {'state': 0, 'l_center': np.array([90., 110.], dtype=np.float32),
                 'r_center': np.array([130., 110.], dtype=np.float32),
                 'eye_size': np.float32(39.0)}
        out = corrector_cpu.correct(crop, crop, None, gf_ok)
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


class TestEyeSourceModes:
    """GAZE_TIMING_PLAN U7-U16：双模式语义 / bypass 不变式 / 守卫 / 别名。"""

    def _gf(self, state=0, eye_size=39.0):
        return {'state': state,
                'l_center': np.array([90., 110.], dtype=np.float32),
                'r_center': np.array([130., 110.], dtype=np.float32),
                'eye_size': np.float32(eye_size)}

    def _pair(self, seed=7):
        g = torch.Generator().manual_seed(seed)
        src = (torch.randn(3, 256, 256, generator=g) * 0.3).clamp(-1, 1)
        canvas = (torch.randn(3, 256, 256, generator=g) * 0.3).clamp(-1, 1)
        return src, canvas

    def test_u11_rendered_bypass_bitwise(self, corrector_cpu):
        """rendered 模式：bypass 帧（眨眼）必须逐位返回 canvas——不做原生
        贴回（回归契约：与改造前行为逐位一致，即使 eye_source != canvas）。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('rendered')
        src, canvas = self._pair()
        out = corrector_cpu.correct(src, canvas, None, self._gf(state=1))
        assert torch.equal(out, canvas)
        out = corrector_cpu.correct(src, canvas, None, self._gf(state=2))
        assert torch.equal(out, canvas)

    def test_u7_ori_blink_pastes_native(self, corrector_cpu):
        """ori 模式 bypass 不变式：眨眼帧贴未修正原生眼（眼中心像素 ==
        eye_source，mask 中心=1），且不跑生成（corrected_frames 不变）。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('ori')
        src, canvas = self._pair()
        before = corrector_cpu.corrected_frames
        out = corrector_cpu.correct(src, canvas, None, self._gf(state=1))
        assert not torch.equal(out, canvas)          # 原生贴回发生了
        assert not torch.equal(out, src)
        assert out[:, 110, 90].isclose(src[:, 110, 90], atol=1e-5).all()
        assert corrector_cpu.corrected_frames == before

    def test_u7_ori_gate_off_pastes_native_no_ghost(self, corrector_cpu):
        """adaptive 门关帧（generate=False）：贴原生眼 + 清生成历史
        （F3：门重开不得 lerp 陈旧 _prev_gen）。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('ori')
        src, canvas = self._pair()
        # 先跑一帧生成，制造 _prev_gen 历史
        corrector_cpu.correct(src, canvas, None, self._gf(state=0), generate=True)
        assert corrector_cpu._prev_gen is not None
        out = corrector_cpu.correct(src, canvas, None, self._gf(state=0),
                                    generate=False)
        assert out[:, 110, 90].isclose(src[:, 110, 90], atol=1e-5).all()
        assert corrector_cpu._prev_gen is None       # 门关即清历史

    def test_u7_ori_normal_two_phase(self, corrector_cpu):
        """ori 模式正常帧：两段都执行（原生贴回 + 生成），输出 != 两输入。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('ori')
        src, canvas = self._pair()
        out = corrector_cpu.correct(src, canvas, None, self._gf(state=0),
                                    generate=True)
        assert not torch.equal(out, canvas)
        assert not torch.equal(out, src)
        assert corrector_cpu.corrected_frames == 1

    def test_u8_canvas_domain_guard(self, corrector_cpu):
        """canvas 域守卫（DX-T2）：形状/数值域非法 → 原样返回 canvas +
        error_skips 计数，不崩溃。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('ori')
        src = (torch.randn(3, 256, 256) * 0.3).clamp(-1, 1)
        bad_shape = torch.randn(3, 128, 256) * 0.3
        out = corrector_cpu.correct(src, bad_shape, None, self._gf(0))
        assert torch.equal(out, bad_shape)
        assert corrector_cpu.error_skips == 1
        bad_range = torch.full((3, 256, 256), 5.0)
        out = corrector_cpu.correct(src, bad_range, None, self._gf(0))
        assert torch.equal(out, bad_range)
        assert corrector_cpu.error_skips == 2

    def test_u9_first_frame_gate_off_facemark_path(self, corrector_cpu):
        """legacy facemark 路径段首即门关（_fixed_eye_size=None 守卫，
        ENG F3）：不崩溃、贴回有效。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('ori')
        src, canvas = self._pair()
        fm = _make_facemark()
        out = corrector_cpu.correct(src, canvas, fm, None, generate=False)
        assert out[:, int(fm[40, 1]), int(fm[40, 0])].isclose(
            src[:, int(fm[40, 1]), int(fm[40, 0])], atol=1e-4).all()
        assert corrector_cpu._fixed_eye_size is not None

    def test_u10_native_paste_channel_fidelity(self, corrector_cpu):
        """通道序（ENG F6）：R≠B 合成帧——原生贴回后眼中心通道值跟随
        eye_source（防 BGR/RGB 互换）。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('ori')
        src = torch.zeros(3, 256, 256)
        src[0, 100:120, 80:100] = 0.8   # R 亮
        src[2, 100:120, 80:100] = -0.8  # B 暗
        canvas = torch.zeros(3, 256, 256)
        canvas[0, 100:120, 80:100] = -0.8
        canvas[2, 100:120, 80:100] = 0.8
        out = corrector_cpu.correct(src, canvas, None, self._gf(state=1))
        assert out[0, 110, 90] > 0.5   # R 跟随 eye_source（亮）
        assert out[2, 110, 90] < -0.5  # B 跟随 eye_source（暗）

    def test_u15_reset_then_idempotent(self, corrector_cpu):
        """幂等断言语义（ENG F4）：(a) eye_source 输入跨渲染逐位不变；
        (b) 两次渲染间 reset() 后输出逐位相等（时序平滑使连续渲染输出
        必然不同——那是 by design，不是缺陷）。"""
        corrector_cpu.reset()
        corrector_cpu.set_eye_source('ori')
        src, canvas = self._pair(11)
        snap = src.clone()
        out1 = corrector_cpu.correct(src, canvas, None, self._gf(0), generate=True)
        assert torch.equal(src, snap)                # (a) 输入不变
        corrector_cpu.reset()
        out2 = corrector_cpu.correct(src, canvas, None, self._gf(0), generate=True)
        assert torch.equal(out1, out2)               # (b) reset 后确定性

    def test_u16_eye_source_validation(self, corrector_cpu):
        """eye_source 拒绝式校验（ENG F9/E-T7）：非法值 raise，合法值切换
        并清生成历史。"""
        corrector_cpu.reset()
        corrector_cpu.correct((torch.randn(3, 256, 256) * 0.3).clamp(-1, 1),
                              (torch.randn(3, 256, 256) * 0.3).clamp(-1, 1),
                              None, self._gf(0), generate=True)
        assert corrector_cpu._prev_gen is not None
        with pytest.raises(ValueError):
            corrector_cpu.set_eye_source('bogus')
        corrector_cpu.set_eye_source('rendered')
        assert corrector_cpu.eye_source_mode == 'rendered'
        assert corrector_cpu._prev_gen is None       # 语义切换清历史

    def test_u14_pristine_no_alias(self):
        """别名防护（ENG F5）：extract_pristine_crop 返回新内存——修改
        orifra 不影响 pristine（blend 原地写 orifra 的既有语义下，眼源
        必须不可被 blend 触达）。"""
        from realtime_serve.sequence_frame_pack import extract_pristine_crop
        import numpy as np
        frame = np.zeros((2560, 1440, 3), dtype=np.uint8)
        frame[1000:1256, 500:756] = 200
        meta = {'crop_x': 500.0, 'crop_y': 1000.0, 'crop_size': 256.0}
        pristine = extract_pristine_crop(frame, meta)
        assert pristine is not None and pristine.shape == (256, 256, 3)
        assert not np.shares_memory(pristine, frame)
        snapshot = pristine.copy()
        frame[1000:1256, 500:756] = 10   # 模拟 blend 原地写
        assert np.array_equal(pristine, snapshot)
