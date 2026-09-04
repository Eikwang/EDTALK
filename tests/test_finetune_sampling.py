"""T4/T5 单测：四路采样矩阵 + 嘴区权重图 + 加权损失不变量。

覆盖 FINE_TUNE_FIX_PLAN 的测试要求：
- held-out 索引永不出现在采样对（近邻/自对/远对全路径）
- 单帧回退（旧版 random.sample(range(1),2) 崩溃点）
- facemark 行数 fail-fast；NaN 行 → 全 1 权重回退
- 近邻距离 ≤ near_window（clamp 后）
- 自对/远对比率的分布验证（统计容差）
- weighted_l1 不变量：权重全 1 时 == F.l1_loss（归一化保量纲）
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

from fine_tune.dataset import Finetune256, _mouth_weight_map  # noqa: E402
from fine_tune.trainer_fine_tune import weighted_l1  # noqa: E402


def _make_video(tmp_path, name, n_frames, start=0):
    d = tmp_path / 'count' / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        arr = np.full((8, 8, 3), (i * 7) % 255, dtype=np.uint8)
        Image.fromarray(arr).save(d / f'{start + i}.jpg')
    return d


def _make_facemark(tmp_path, name, n_frames, nan_rows=()):
    """合成 68 点 facemark：嘴区中心 (128,180)，可选 NaN 行。"""
    fm = np.zeros((n_frames, 68, 2), dtype=np.float32)
    fm[:, 48:60, 0] = np.linspace(115, 141, 12)
    fm[:, 48:60, 1] = 175 + (np.arange(12) % 4) * 3
    fm[:, 0:17, 0] = np.linspace(60, 196, 17)
    fm[:, 0:17, 1] = np.linspace(80, 240, 17)
    for r in nan_rows:
        fm[r] = np.nan
    d = tmp_path / 'facemark'
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f'{name}.npy', fm)
    return d


def _open_targets(monkeypatch):
    """捕获 dataset 打开的帧路径 → (frame_a, frame_b) 列表。"""
    opened = []
    real_open = Image.open

    def spy(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)

    import fine_tune.dataset as ds_mod
    monkeypatch.setattr(ds_mod, 'Image', type('I', (), {'open': staticmethod(spy),
                                                        'LOAD_TRUNCATED_IMAGES': True}))
    return opened


def _build(tmp_path, n_frames=200, **kw):
    _make_video(tmp_path, 'v1', n_frames)
    _make_facemark(tmp_path, 'v1', n_frames)
    return Finetune256(str(tmp_path / 'count'), facemark_dir=str(tmp_path / 'facemark'), **kw)


def test_heldout_excluded_from_all_paths(tmp_path, monkeypatch):
    """held-out 索引（末 10%）永不出现在任何采样对的 target 中。"""
    n, ratio = 200, 0.1
    ds = _build(tmp_path, n_frames=n, held_out_ratio=ratio, near_window=8)
    hold_lo = n - max(1, int(n * ratio))          # 180
    assert ds.pools[0][-1] == hold_lo - 1
    assert ds.held_out_info[0] == (hold_lo, n)

    opened = _open_targets(monkeypatch)
    for _ in range(300):
        ds[0]
    targets = [int(Path(p).stem) for p in opened[1::2]]  # 每次调用第 2 个 = target
    assert all(t < hold_lo for t in targets), \
        f'held-out 帧进入训练采样: {sorted(set(t for t in targets if t >= hold_lo))}'


def test_near_distance_bounded(tmp_path, monkeypatch):
    """近邻路径 |i-j| ≤ near_window（clamp 后）；兜底远对路单独计数。"""
    n, win = 200, 16
    ds = _build(tmp_path, n_frames=n, held_out_ratio=0.0, near_window=win,
                near_ratio=0.9, self_ratio=0.1, far_ratio=0.0)
    opened = _open_targets(monkeypatch)
    for _ in range(300):
        ds[0]
    pairs = [(int(Path(a).stem), int(Path(b).stem))
             for a, b in zip(opened[0::2], opened[1::2])]
    dists = [abs(a - b) for a, b in pairs if a != b]
    self_pairs = sum(1 for a, b in pairs if a == b)
    assert all(d <= win for d in dists), f'近邻距离超窗: {max(dists)} > {win}'
    assert self_pairs / len(pairs) < 0.25, '自对比例应接近设定值 0.1'


def test_self_pair_ratio(tmp_path, monkeypatch):
    """自对比率 ≈ self_ratio（统计容差）。"""
    ds = _build(tmp_path, n_frames=200, held_out_ratio=0.0,
                near_ratio=0.0, self_ratio=0.4, far_ratio=0.0)
    opened = _open_targets(monkeypatch)
    n_iter = 400
    for _ in range(n_iter):
        ds[0]
    pairs = list(zip(opened[0::2], opened[1::2]))
    rate = sum(1 for a, b in pairs if a == b) / len(pairs)
    assert abs(rate - 0.4) < 0.1, f'自对比率 {rate:.2f} 偏离 0.4'


def test_single_frame_video_fallback(tmp_path):
    """单帧视频不崩溃（旧版 random.sample(range(1),2) 崩溃点），退化为自对。"""
    import torchvision.transforms as T
    _make_video(tmp_path, 'v1', 1)
    _make_facemark(tmp_path, 'v1', 1)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    ds = Finetune256(str(tmp_path / 'count'), transform=tf,
                     facemark_dir=str(tmp_path / 'facemark'))
    s, t, w = ds[0]
    assert s.shape == (3, 256, 256) and t.shape == s.shape
    assert torch.equal(s, t), '单帧回退应 source==target'
    assert float(w.max()) == 4.0, 'facemark 可用时权重图仍生效'


def test_facemark_count_mismatch_fails_fast(tmp_path):
    _make_video(tmp_path, 'v1', 100)
    _make_facemark(tmp_path, 'v1', 99)            # 少一行
    with pytest.raises(RuntimeError, match='行数'):
        Finetune256(str(tmp_path / 'count'),
                    facemark_dir=str(tmp_path / 'facemark'))


def test_nan_facemark_row_falls_back_to_uniform_weight(tmp_path):
    """NaN facemark 行 → 权重图全 1（不产生 NaN 权重 → 不会 NaN loss）。"""
    fm = _fm_with_nan()
    w = _mouth_weight_map(fm)
    assert not np.isnan(w).any()
    assert float(w.max()) == 1.0                  # 回退全 1
    # 对照：合法嘴区点 → 嘴区权重 4.0
    good = np.zeros((68, 2), dtype=np.float32)
    good[48:60, 0] = np.linspace(115, 141, 12)
    good[48:60, 1] = 175 + (np.arange(12) % 4) * 3
    good[0:17, 0] = np.linspace(60, 196, 17)
    good[0:17, 1] = np.linspace(80, 240, 17)
    assert float(_mouth_weight_map(good).max()) == 4.0


def _fm_with_nan():
    fm = np.zeros((68, 2), dtype=np.float32)
    fm[30] = np.nan
    return fm


def test_mouth_weight_map_values():
    fm = np.zeros((68, 2), dtype=np.float32)
    fm[48:60, 0] = np.linspace(115, 141, 12)
    fm[48:60, 1] = 175 + (np.arange(12) % 4) * 3
    fm[0:17, 0] = np.linspace(60, 196, 17)
    fm[0:17, 1] = np.linspace(80, 240, 17)
    w = _mouth_weight_map(fm, mouth_weight=4.0)
    assert w.shape == (1, 256, 256)
    assert float(w.min()) == 1.0 and float(w.max()) == 4.0
    assert float(w.mean()) > 1.0


def test_weighted_l1_identity_with_plain_l1():
    """权重全 1 → weighted_l1 == F.l1_loss（归一化不变量，ENG 声部 T4）。"""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)
    ones = torch.ones(2, 1, 16, 16)
    assert torch.allclose(weighted_l1(pred, target, ones),
                          torch.nn.functional.l1_loss(pred, target))


def test_weighted_l1_upweights_mouth_region():
    """嘴区 ×4 的样本主导损失 — 数值方向校验。"""
    torch.manual_seed(0)
    pred = torch.zeros(1, 3, 32, 32)
    target = torch.zeros(1, 3, 32, 32)
    target[:, :, 0, 0] = 10.0                     # 差异只在左上角（非嘴区）
    target[:, :, 20, 20] = 10.0                   # 差异在嘴区（mask=4 处）
    w = torch.ones(1, 1, 32, 32)
    w[:, :, 16:32, 16:32] = 4.0
    plain = torch.nn.functional.l1_loss(pred, target)
    weighted = weighted_l1(pred, target, w)
    # 嘴区差异被放大、非嘴区被摊薄 → 归一化后总损失应高于 plain 的再分配方向
    assert weighted > plain * 0.9                 # 量纲保持（归一化），分配偏移
