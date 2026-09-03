# -*- coding: utf-8 -*-
"""步骤 11: 视线纠正数据预处理 -> gaze_data/

从已有 facemark/<segment>.npy（68 点，256 空间）推导实时视线纠正所需的
时序数据，输出 gaze_data/<segment>.npy。实时推理循环查表使用，不再在
每帧上实时计算 EAR / 眼中心（256 空间单帧关键点噪声 ±0.5-1px，会导致
纠正框抖动与眨眼漏检）。

输出结构（np.save 的 dict）:
    state        (N,)    uint8   0=正常纠正 1=眨眼中(跳过) 2=无数据(跳过)
    l_center     (N,2)   float32 左眼中心 256 空间（零相位平滑）
    r_center     (N,2)   float32 右眼中心 256 空间（零相位平滑）
    eye_size     ()      float32 段级锁定纠正框尺寸（有效帧中位数）
    ear          (N,)    float32 原始 EAR（时序中值滤波后，供阈值校准）

用法:
    # 单个角色资产目录（含 facemark/ 与 metadata/ 子目录，如 wy）
    python data_preprocess/data_preprocess_for_train/prepare_gaze_data.py --asset_dir wy

    # 数据集根目录（扫描含 facemark/ 的子目录，与 run_preprocess 步骤 11 一致）
    python data_preprocess/data_preprocess_for_train/prepare_gaze_data.py --base_dir HDTF
"""

import os
import sys
import argparse

import numpy as np

# 眨眼判定不再使用全局固定阈值（EAR_CLOSE/EAR_OPEN）——固定阈值在头部
# 俯仰时失配（256 空间眼睛整体变窄、EAR 基线漂移）。改用相对本段中位数
# 的自适应阈值，见 mark_skip_frames()。

L_EYE = list(range(36, 42))
R_EYE = list(range(42, 48))


def ear_of(pts: np.ndarray) -> float:
    """Eye aspect ratio（与实时实现同一公式）。"""
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return float((v1 + v2) / (2.0 * h))


def median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    """一维中值滤波（边界复制）。NaN 由调用方预先插值。"""
    if k < 2:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(windows, axis=-1)


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """前向 EMA。"""
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = alpha * x[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def smooth_zero_phase(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """零相位平滑：前向 EMA + 反向 EMA 取平均（不引入实时路径的滞后）。"""
    fwd = ema(x, alpha)
    bwd = ema(x[::-1], alpha)[::-1]
    return (fwd + bwd) / 2.0


def interp_nan(x: np.ndarray) -> np.ndarray:
    """线性插值 NaN（全 NaN 序列原样返回）。"""
    out = x.astype(np.float64).copy()
    nans = np.isnan(out)
    if not nans.any() or nans.all():
        return out.astype(np.float32)
    idx = np.arange(len(out))
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
    return out.astype(np.float32)


def remove_short_runs(state: np.ndarray, min_len: int = 3) -> np.ndarray:
    """移除短于 min_len 的游程（合并到前邻），防止单帧修正/跳过交替。"""
    s = state.copy()
    n = len(s)
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        run = j - i
        if 0 < i and j <= n and run < min_len:
            s[i:j] = s[i - 1]
        i = j
    return s


def mark_skip_frames(ear_s: np.ndarray, skip_ratio: float = 0.6):
    """标记"不适合施加视线纠正"的帧。

    判定是相对本段睁眼度的自适应阈值：
        skip_th = skip_ratio × median(ear_s)     （默认 0.6）
    相对比例吸收头部俯仰造成的 EAR 基线漂移。

    0.6 的取值权衡（用户实测校准）：
    - 往下看（眼睑自然下垂，EAR ≈ 中位数的 0.5-0.7）应继续纠正
      ——瞳孔清晰可见，纠正有效且符合预期；
    - 真眨眼的深闭期（EAR < 0.45×中位）仍被跳过，避免生成假眼；
    - 深度微眯（EAR < 0.5×中位）被跳过；浅微眯可能被纠正，实测可接受。
    偏高（如 0.75）会把正常往下看误判为眨眼并跳过纠正。

    Returns:
        (state, skip_th): state 1=跳过 0=正常；skip_th 为实际使用阈值。
    """
    med = float(np.median(ear_s))
    skip_th = max(0.06, skip_ratio * med)
    state = (ear_s < skip_th).astype(np.uint8)
    state = remove_short_runs(state, min_len=3)
    return state, skip_th


def process_asset(asset_dir: str, skip_ratio: float = 0.6,
                  smooth_alpha: float = 0.15) -> bool:
    facemark_dir = os.path.join(asset_dir, "facemark")
    if not os.path.isdir(facemark_dir):
        return False
    out_dir = os.path.join(asset_dir, "gaze_data")
    os.makedirs(out_dir, exist_ok=True)

    segments = sorted(f for f in os.listdir(facemark_dir) if f.endswith(".npy"))
    if not segments:
        print(f"  [WARN] {facemark_dir} 下无 npy 文件")
        return False

    for seg_file in segments:
        seg_name = os.path.splitext(seg_file)[0]
        in_path = os.path.join(facemark_dir, seg_file)
        out_path = os.path.join(out_dir, seg_file)

        fm = np.load(in_path)
        n = fm.shape[0]

        # 1) 原始 EAR（任一眼闭即视为闭，与实时 min_ear 一致）
        ear_raw = np.array([
            min(ear_of(fm[i, L_EYE]), ear_of(fm[i, R_EYE]))
            if not np.isnan(fm[i]).any() else np.nan
            for i in range(n)
        ], dtype=np.float32)

        # 2) NaN 帧（检测失败）先行标记；EAR 时序插值 + 中值滤波去量化噪声
        nan_mask = np.isnan(ear_raw)
        ear_filled = interp_nan(ear_raw)
        ear_s = median_filter(ear_filled, 7)

        # 3) 跳过帧标记（相对睁眼度自适应阈值）
        state, skip_th = mark_skip_frames(ear_s, skip_ratio=skip_ratio)
        state[nan_mask] = 2  # 无数据帧跳过纠正

        # 4) 眼中心：外包络中心 → NaN 插值 → 中值滤波 → 零相位平滑
        def center_series(ids):
            pts = fm[:, ids, :]  # (N, 6, 2)
            mn = np.nanmin(pts, axis=1)
            mx = np.nanmax(pts, axis=1)
            c = (mn + mx) / 2.0  # (N,2)
            c[:, 0] = interp_nan(c[:, 0])
            c[:, 1] = interp_nan(c[:, 1])
            c[:, 0] = smooth_zero_phase(median_filter(c[:, 0], 7), smooth_alpha)
            c[:, 1] = smooth_zero_phase(median_filter(c[:, 1], 7), smooth_alpha)
            return c.astype(np.float32)

        l_center = center_series(L_EYE)
        r_center = center_series(R_EYE)

        # 5) 段级 eye_size：单眼 1.5*max(w,h) 的中位数（与实时锁定路径同口径）
        sizes = []
        for i in range(n):
            if nan_mask[i]:
                continue
            for ids in (L_EYE, R_EYE):
                w = np.nanmax(fm[i, ids, 0]) - np.nanmin(fm[i, ids, 0])
                h = np.nanmax(fm[i, ids, 1]) - np.nanmin(fm[i, ids, 1])
                sizes.append(1.5 * max(w, h))
        eye_size = float(np.clip(np.median(sizes), 16.0, 128.0)) if sizes else 48.0

        np.save(out_path, {
            "state": state,
            "l_center": l_center,
            "r_center": r_center,
            "eye_size": np.float32(eye_size),
            "ear": ear_s.astype(np.float32),
        })

        blink_pct = 100.0 * (state == 1).sum() / max(n, 1)
        nan_pct = 100.0 * nan_mask.sum() / max(n, 1)
        print(f"  [{seg_name}] frames={n} skip={blink_pct:.1f}% "
              f"no_data={nan_pct:.1f}% eye_size={eye_size:.1f} "
              f"ear_med={float(np.median(ear_s)):.3f} skip_th={skip_th:.3f}")
    return True


def find_assets(base: str):
    """base 本身是资产目录，或扫描其下含 facemark/ 的一级子目录。"""
    if os.path.isdir(os.path.join(base, "facemark")):
        return [base]
    assets = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(os.path.join(p, "facemark")):
            assets.append(p)
    return assets


def main():
    parser = argparse.ArgumentParser(description="步骤 11: 视线纠正数据预处理")
    parser.add_argument("--base_dir", default=None,
                        help="数据集根目录（扫描含 facemark/ 的子目录）")
    parser.add_argument("--asset_dir", default=None,
                        help="单角色资产目录（直接含 facemark/，如 wy）")
    parser.add_argument("--skip_ratio", type=float, default=0.6,
                        help="跳过纠正的相对睁眼度阈值（×段中位 EAR）。越低越少跳过："
                             "0.6=往下看仍纠正、深眨眼跳过；调高（如 0.75）会更早跳过微眯")
    parser.add_argument("--smooth_alpha", type=float, default=0.15,
                        help="眼中心零相位 EMA 系数（越小越平滑）")
    args = parser.parse_args()

    target = args.asset_dir or args.base_dir
    if not target:
        parser.error("需要 --asset_dir 或 --base_dir 之一")
    if not os.path.isdir(target):
        print(f"[ERROR] 目录不存在: {target}")
        sys.exit(1)

    assets = find_assets(target)
    if not assets:
        print(f"[ERROR] 未找到含 facemark/ 的资产目录: {target}")
        sys.exit(1)

    print(f"[gaze_data] 处理 {len(assets)} 个资产目录")
    failed = []
    for a in assets:
        print(f"[gaze_data] asset: {a}")
        if not process_asset(a, skip_ratio=args.skip_ratio,
                             smooth_alpha=args.smooth_alpha):
            failed.append(a)
            print(f"  [FAIL] {a}")
    if failed:
        print(f"[gaze_data] {len(failed)} 个资产失败: {failed}")
        sys.exit(1)
    print("[gaze_data] 完成")
    sys.exit(0)


if __name__ == "__main__":
    main()
