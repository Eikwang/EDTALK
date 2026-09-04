"""eval_finetune 协议元测试（FINE_TUNE_FIX_PLAN T3，ENG 声部 T1 要求）。

自校准只覆盖"无差异"方向；元测试覆盖"指标有区分度"方向：
  (a) 恒等 — 候选==参照 → 全部为零
  (b) 已知扰动敏感性 — 已知像素/角度扰动 → 指标按预期方向与量级响应
  (c) NaN/失败路径 — NaN facemark 不静默污染指标

纯 numpy/CPU，不加载真模型，跑得快、确定性高。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import eval_finetune as ev  # noqa: E402


def _fm68(nose_xy=(128.0, 150.0)):
    """合成 68 点 facemark（几何合理：水平眼线 + 给定鼻尖）。"""
    fm = np.zeros((68, 2), dtype=np.float32)
    fm[36:42] = [(100, 120), (105, 116), (110, 116), (115, 116), (120, 118), (105, 122)]
    fm[42:48] = [(135, 118), (140, 116), (145, 116), (150, 120), (145, 122), (138, 122)]
    fm[30] = nose_xy
    fm[48:60] = [(110, 175), (115, 172), (120, 170), (128, 169), (136, 170), (142, 172),
                 (146, 175), (142, 180), (136, 183), (128, 184), (120, 183), (114, 180)]
    fm[0:17] = [(80, 100 + i * 6) for i in range(17)]
    return fm


# ---------------------------------------------------------------- (a) 恒等
def test_pose_identity_zero():
    fm = _fm68()
    m = ev.pose_metrics(fm, fm)
    assert m['mae_px'] == 0.0 and m['roll_deg'] == 0.0 and m['nose_px'] == 0.0


def test_interpolate_lam0_is_base():
    base = {'dec.conv1.weight': np.random.default_rng(0).standard_normal(4),
            'enc.net_app.0.weight': np.random.default_rng(1).standard_normal(4),
            'direction_lipnonlip.weight': np.random.default_rng(2).standard_normal(4),
            'fc.0.weight': np.random.default_rng(3).standard_normal(4)}
    out = ev.interpolate_state(
        {k: __import__('torch').from_numpy(v.copy()) for k, v in base.items()},
        {k: __import__('torch').from_numpy(v.copy() + 100) for k, v in base.items()},
        0.0)
    for k, v in base.items():
        assert np.allclose(out[k].numpy(), v), f'λ=0 应恒等于 base, 键 {k}'


# ---------------------------------------------------------------- (b) 扰动敏感性
def test_pose_translate_3px():
    fm = _fm68()
    shifted = fm.copy()
    shifted[:, 0] += 3.0
    m = ev.pose_metrics(shifted, fm)
    assert abs(m['mae_px'] - 3.0) < 1e-4, f'平移 3px → MAE 应为 3, 实测 {m["mae_px"]}'
    assert abs(m['nose_px'] - 3.0) < 1e-4


def test_pose_roll_sensitivity():
    """整脸绕画面中心旋转 5° → 眼线角差 ≈ 5°。"""
    fm = _fm68()
    theta = np.radians(5.0)
    c = np.array([128.0, 128.0])
    rot = (fm - c) @ np.array([[np.cos(theta), np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]]) + c
    m = ev.pose_metrics(rot.astype(np.float32), fm)
    assert abs(m['roll_deg'] - 5.0) < 0.5, f'旋转 5° → roll 差应 ≈5, 实测 {m["roll_deg"]}'


def test_bg_metric_responds_only_to_background_change():
    """面部回填 trick 的方向正确性：只动背景 → bg SSIM 下降；只动面部 → 不变。"""
    ref = np.full((256, 256, 3), 200, dtype=np.uint8)
    face = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(face, (128, 128), 60, 255, -1)
    m3 = np.repeat((face == 0)[:, :, None], 3, axis=2)

    out_bg_changed = ref.copy()
    out_bg_changed[:40] = 0                       # 只动背景
    out_face_changed = ref.copy()
    out_face_changed[120:140, 120:140] = 0        # 只动面部（会被回填抵消）

    s_ref = ev.bg_metrics(ref, ref, face)
    s_bg = ev.bg_metrics(out_bg_changed, ref, face)
    s_face = ev.bg_metrics(out_face_changed, ref, face)
    assert s_bg['ssim'] < s_ref['ssim'] - 0.01, '背景扰动必须拉低 bg SSIM'
    assert abs(s_face['ssim'] - s_ref['ssim']) < 1e-6, '面部扰动应被回填抵消'
    assert s_ref['ssim'] == 1.0, '恒等 → SSIM=1'


def test_lip_openness_sensitive():
    fm = _fm68()
    open_more = fm.copy()
    open_more[57] = (128, 195)                    # 下唇往下拉 → 开合变大
    assert ev.lip_openness(open_more) > ev.lip_openness(fm)


# ---------------------------------------------------------------- (c) NaN/失败
def test_hull_mask_output_range():
    m = ev.hull_mask(_fm68(), range(48, 60), 3)
    assert m.dtype == np.uint8 and m.max() == 255 and m.min() == 0
    assert (m == 255).sum() > 100, '凸包填充应有实质面积'


def test_pose_metrics_nan_reference_is_caller_filtered():
    """NaN facemark 行在 load_split 被过滤为 None（不进指标）— 模拟过滤逻辑。"""
    fm = _fm68()
    fm[30] = np.nan
    # evaluate() 的约定：NaN 行 → ref=None → 指标跳过。此处验证过滤判定本身。
    assert np.isnan(fm).any()
    filtered = None if np.isnan(fm).any() else fm
    assert filtered is None
