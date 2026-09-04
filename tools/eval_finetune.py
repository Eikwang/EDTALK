"""微调评估协议（FINE_TUNE_FIX_PLAN T3 / 方案 D）。

对底模与候选（微调）ckpt 在 held-out 帧上跑四组指标，输出 JSONL 归档：
  1. pose_keep    — 姿势保持：渲染帧 68 点 landmark vs 原帧预计算 facemark
                    （全点 MAE px + 眼线 roll 角差 deg + 鼻尖偏移 px）
  2. lip_fidelity — 同一 lip alpha 过底模/候选：嘴区像素 MAE + 开合幅度比
                    （还原测试：alpha 从帧自身提取，与实时推理 render_loop.py:727
                    的 pose drive=source 同条件）
  3. background   — 面部凸包外区域 SSIM/PSNR（渲染 vs 原帧；诊断指标，
                    底模自身得分即可达上限参照）
  4. paste_proxy  — 嘴区 landmark MAE（贴回错位的直接代理；T7 落地后
                    接入 compositor 完整 E2E）

设计要点（评审吸收）：
- held-out = 每视频末 10%（时间切分；局限：与训练帧相邻，泛化偏乐观，
  对本用例推理分布一致 — 结果 JSON 中带 held_out_note 标注）
- 同帧底模输出缓存 → lip 指标公平对比
- 检测失败帧 skip + 计入 det_fail_rate（不静默）
- 确定性：seed 固定采样 / eval / inference_mode / cudnn.benchmark=False
- --sweep: 分组插值扫描（dec/enc 插值，direction/fc 系保持底模 —
  direction 的 QR 分解使全量插值非线性，不插）

运行示例：
  python tools/eval_finetune.py --cand-ckpt ckpt_models/wy1/checkpoint/020180.pt
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from networks.generator_lip_pose import Generator  # noqa: E402

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_LANDMARK_DAT = REPO_ROOT / 'ckpts' / 'shape_predictor_68_face_landmarks.dat'


# ---------------------------------------------------------------- 数据
def _frame_sort_key(p):
    try:
        return (0, int(Path(p).stem))
    except ValueError:
        return (1, Path(p).stem)


def list_videos(datapath):
    """返回 {video_name: [sorted frame paths]} — 与 Finetune256 同结构约定。"""
    datapath = Path(datapath)
    videos = {}
    for sub in sorted(p for p in datapath.iterdir() if p.is_dir()):
        frames = [str(p) for p in sub.iterdir()
                  if p.suffix.lower() in IMG_EXTS]
        if frames:
            videos[sub.name] = sorted(frames, key=_frame_sort_key)
    return videos


def load_split(videos, held_out_ratio, split, num_frames, seed, facemark_root):
    """按视频采样评估帧；held-out = 每视频末 held_out_ratio 比例。

    返回 [(video, frame_path, facemark(68,2)float32 or None)]。
    """
    rng = np.random.default_rng(seed)
    samples = []
    for name, frames in videos.items():
        facemark_npy = Path(facemark_root) / f'{name}.npy'
        fm = np.load(facemark_npy) if facemark_npy.exists() else None
        if fm is not None and len(fm) != len(frames):
            raise RuntimeError(
                f'[{name}] facemark 行数 {len(fm)} != 帧数 {len(frames)} — '
                f'缓存与帧序错位 (fail-fast, 计划 A2)')
        n = len(frames)
        hold_n = max(1, int(n * held_out_ratio))
        lo = n - hold_n if split == 'heldout' else 0
        hi = n if split == 'heldout' else n - hold_n
        pool = list(range(lo, hi))
        take = min(num_frames, len(pool))
        idxs = sorted(rng.choice(pool, size=take, replace=False).tolist())
        for i in idxs:
            ref = None if fm is None else fm[i].copy()
            # NaN facemark 行（检测失败帧）→ 视为无参照，跳过 pose 系指标
            if ref is not None and np.isnan(ref).any():
                ref = None
            samples.append((name, frames[i], ref))
    return samples


def frame_to_tensor(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                       # [-1,1]，与训练/推理一致
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def tensor_to_bgr(t):
    """[-1,1] tensor → uint8 BGR（dlib 检测与 skimage 指标用）。"""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1) / 2 * 255).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 模型
def load_generator(ckpt_path, device):
    gen = Generator().to(device)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    gen.load_state_dict(ck['gen'])                # strict — 结构漂移即报错
    gen.eval()
    return gen


def infer_frame(gen, img, alpha_lip):
    with torch.inference_mode():
        return gen.test_from_audio_pose_image(img, alpha_lip, img)  # pose drive=source 自身


def extract_lip_alpha(gen, img):
    """从帧自身提取 lip alpha（还原测试条件）。"""
    with torch.inference_mode():
        return gen.get_lip_pose_feature(img)[0]   # (1,20)


# ---------------------------------------------------------------- landmark
def get_predictor(dat_path=DEFAULT_LANDMARK_DAT):
    import dlib
    return dlib.shape_predictor(str(dat_path))


def detect_landmarks_bgr(predictor, bgr):
    """渲染输出帧的 68 点 landmark（256 空间，全图 bbox）。失败返回 None。"""
    try:
        rect = dlib.rectangle(0, 0, bgr.shape[1], bgr.shape[0])
        shape = predictor(bgr, rect)
        pts = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            pts[i, 0] = shape.part(i).x
            pts[i, 1] = shape.part(i).y
        return pts
    except Exception:
        return None


# ---------------------------------------------------------------- 指标
def hull_mask(fm68, indices, dilate_px):
    """landmark 子集凸包填充 mask（256×256 uint8, 0/255）。"""
    m = np.zeros((256, 256), dtype=np.uint8)
    pts = fm68[list(indices)].astype(np.int32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m


def pose_metrics(pred_fm, ref_fm):
    """姿势保持：全点 MAE px + 眼线 roll 角差 deg + 鼻尖偏移 px。"""
    mae = float(np.mean(np.linalg.norm(pred_fm - ref_fm, axis=1)))
    def eye_angle(fm):
        """眼线角（roll）：左眼中心(36-41) → 右眼中心(42-47) 连线与水平轴夹角。"""
        lx, ly = fm[36:42, 0].mean(), fm[36:42, 1].mean()
        rx, ry = fm[42:48, 0].mean(), fm[42:48, 1].mean()
        return float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    roll = eye_angle(pred_fm) - eye_angle(ref_fm)
    nose = float(np.linalg.norm(pred_fm[30] - ref_fm[30]))
    return {'mae_px': mae, 'roll_deg': abs(float(roll)), 'nose_px': nose}


def lip_openness(fm68):
    """嘴开合度：上唇顶(51)与下唇底(57)的距离 px。"""
    return float(np.linalg.norm(fm68[57] - fm68[51]))


def bg_metrics(out_bgr, ref_bgr, face_mask):
    """背景 SSIM/PSNR：面部区域用原帧回填 → 差异全部来自背景（标准可解释）。"""
    inv = (face_mask == 0)
    m3 = np.repeat(inv[:, :, None], 3, axis=2)
    out_bg = np.where(m3, out_bgr, ref_bgr).astype(np.uint8)
    ref_bgr_u8 = ref_bgr.astype(np.uint8)
    ssim = float(structural_similarity(out_bg, ref_bgr_u8, channel_axis=2,
                                       data_range=255))
    psnr = float(peak_signal_noise_ratio(ref_bgr_u8, out_bg, data_range=255))
    return {'ssim': ssim, 'psnr': psnr}


# ---------------------------------------------------------------- 插值
def interpolate_state(base_sd, cand_sd, lam):
    """分组插值：dec/enc 线性插值；direction_lipnonlip/fc/lip_fc/pose_fc 保持底模。

    direction 的 QR 分解使 weight 空间插值在函数空间非线性（Q 列可翻转），
    全量插值的 λ 曲线可能非单调 — 故只插 {dec., enc.} 前缀。
    """
    out = {}
    for k, v_base in base_sd.items():
        if (k.startswith('dec.') or k.startswith('enc.')) and k in cand_sd:
            out[k] = (1.0 - lam) * v_base.float() + lam * cand_sd[k].float()
        else:
            out[k] = v_base.clone()
    return out


# ---------------------------------------------------------------- 主评估
def evaluate(gen, predictor, samples, device, base_outputs=None):
    """对采样帧跑指标。base_outputs: {frame_path: uint8 HWC RGB}（lip 对比参照）。

    返回 (per_video 聚合 dict, det_fail_rate)。
    """
    per_frame = {k: [] for k in ('pose_mae', 'pose_roll', 'pose_nose',
                                 'lip_mae', 'lip_open_ratio',
                                 'bg_ssim', 'bg_psnr', 'paste_mae')}
    det_fail = 0
    det_total = 0
    for video, fpath, ref_fm in samples:
        img = frame_to_tensor(fpath, device)
        out = infer_frame(gen, img, extract_lip_alpha(gen, img))
        out_bgr = tensor_to_bgr(out)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        # 姿势保持 + 贴回代理（需要渲染帧 landmark）
        det_total += 1
        pred_fm = detect_landmarks_bgr(predictor, out_bgr)
        if pred_fm is None or ref_fm is None:
            det_fail += 1
        else:
            pm = pose_metrics(pred_fm, ref_fm)
            per_frame['pose_mae'].append(pm['mae_px'])
            per_frame['pose_roll'].append(pm['roll_deg'])
            per_frame['pose_nose'].append(pm['nose_px'])
            mouth_ref = hull_mask(ref_fm, range(48, 60), 3) == 255
            per_frame['paste_mae'].append(
                float(np.mean(np.linalg.norm(
                    (pred_fm - ref_fm)[list(range(48, 60))], axis=1))))
            # 开合比（渲染帧自身检测的嘴距 / 原帧嘴距 — 自还原保真）
            per_frame.setdefault('_open', []).append(
                (lip_openness(pred_fm), lip_openness(ref_fm)))

        # 背景指标（vs 原帧）
        ref_bgr = cv2.cvtColor(
            (np.asarray(Image.open(fpath).convert('RGB'))), cv2.COLOR_RGB2BGR)
        if ref_fm is not None:
            face = hull_mask(ref_fm, range(68), 6)
            bg = bg_metrics(out_bgr, ref_bgr, face)
            per_frame['bg_ssim'].append(bg['ssim'])
            per_frame['bg_psnr'].append(bg['psnr'])

        # lip 像素差（vs 底模同帧输出，同 alpha）
        if base_outputs is not None and fpath in base_outputs:
            ref_out = base_outputs[fpath]
            if ref_fm is not None:
                mouth = hull_mask(ref_fm, range(48, 60), 3)
                diff = np.abs(out_rgb.astype(np.int16)
                              - ref_out.astype(np.int16))[mouth == 255]
                per_frame['lip_mae'].append(float(diff.mean()))
        del img, out, out_bgr, out_rgb

    def agg(key):
        v = per_frame[key]
        return round(float(np.mean(v)), 6) if v else None

    # 开合比聚合（渲染开合 / 原帧开合）
    opens = per_frame.pop('_open', [])
    open_ratio = (round(float(np.mean([o / r for o, r in opens if r > 1e-3])), 4)
                  if opens else None)

    metrics = {
        'pose_keep_mae_px': agg('pose_mae'),
        'pose_keep_roll_deg': agg('pose_roll'),
        'pose_keep_nose_px': agg('pose_nose'),
        'lip_mae_vs_base': agg('lip_mae'),
        'lip_open_ratio_vs_ref': open_ratio,
        'bg_ssim': agg('bg_ssim'),
        'bg_psnr_db': agg('bg_psnr'),
        'paste_proxy_mae_px': agg('paste_mae'),
        'det_fail_rate': round(det_fail / max(det_total, 1), 4),
        'n_frames': det_total,
    }
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base-ckpt', default='ckpts/EDTalk_lip_pose.pt')
    ap.add_argument('--cand-ckpt', nargs='+', required=True,
                    help='候选 ckpt（可多个，如 wy1 各训练步数）')
    ap.add_argument('--datapath', default='wy')
    ap.add_argument('--held-out-ratio', type=float, default=0.1)
    ap.add_argument('--split', choices=['heldout', 'train'], default='heldout')
    ap.add_argument('--num-frames', type=int, default=100,
                    help='每视频采样评估帧数')
    ap.add_argument('--seed', type=int, default=20260904)
    ap.add_argument('--sweep', action='store_true',
                    help='对最后一个候选做分组插值扫描 λ∈{0,.25,.5,.75,1}')
    ap.add_argument('--out-dir', default='tools/eval_results')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f'==> device: {device}')
    if device.type == 'cpu':
        print('[WARN] CPU 模式 — 慢，但指标有效（确定性同样成立）')

    datapath = Path(args.datapath)
    count_dir = datapath / 'count' if (datapath / 'count').is_dir() else datapath
    facemark_root = (datapath / 'facemark' if (datapath / 'facemark').is_dir()
                     else count_dir.parent / 'facemark')
    videos = list_videos(count_dir)
    total = sum(len(v) for v in videos.values())
    print(f'==> {len(videos)} videos, {total} frames; '
          f'split={args.split} (held-out 末 {args.held_out_ratio:.0%}), '
          f'{args.num_frames}/video')
    samples = load_split(videos, args.held_out_ratio, args.split,
                         args.num_frames, args.seed, facemark_root)
    predictor = get_predictor()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / 'eval_results.jsonl'

    print('==> base:', args.base_ckpt)
    base = load_generator(args.base_ckpt, device)
    t0 = time.time()
    base_metrics = evaluate(base, predictor, samples, device)
    base_metrics['wall_s'] = round(time.time() - t0, 1)
    print('    ', json.dumps(base_metrics, ensure_ascii=False))

    # 底模输出缓存（lip 对比参照，uint8 省 4x 内存）
    base_outputs = {}
    for video, fpath, _ in samples:
        img = frame_to_tensor(fpath, device)
        base_outputs[fpath] = cv2.cvtColor(
            tensor_to_bgr(infer_frame(base, img, extract_lip_alpha(base, img))),
            cv2.COLOR_BGR2RGB)

    def append_row(row):
        with open(jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    env = {'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}
    common = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'base_ckpt': args.base_ckpt, 'split': args.split,
              'held_out_ratio': args.held_out_ratio, 'num_frames': args.num_frames,
              'seed': args.seed, 'env': env,
              'held_out_note': '时间切分与训练帧相邻，泛化偏乐观（与本用例推理分布一致）'}

    append_row({**common, 'ckpt': args.base_ckpt, 'role': 'base',
                'metrics': base_metrics})

    for cand_path in args.cand_ckpt:
        print('==> cand:', cand_path)
        cand = load_generator(cand_path, device)
        t0 = time.time()
        m = evaluate(cand, predictor, samples, device, base_outputs=base_outputs)
        m['wall_s'] = round(time.time() - t0, 1)
        print('    ', json.dumps(m, ensure_ascii=False))
        append_row({**common, 'ckpt': cand_path, 'role': 'candidate',
                    'metrics': m})
        del cand
        torch.cuda.empty_cache()

    if args.sweep:
        cand_path = args.cand_ckpt[-1]
        cand_sd = torch.load(cand_path, map_location='cpu',
                             weights_only=False)['gen']
        base_sd = torch.load(args.base_ckpt, map_location='cpu',
                             weights_only=False)['gen']
        sweep_samples = samples[:max(1, len(samples) // 3)]  # 1/3 帧控时长
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            gen = Generator().to(device)
            gen.load_state_dict(interpolate_state(base_sd, cand_sd, lam))
            gen.eval()
            m = evaluate(gen, predictor, sweep_samples, device,
                         base_outputs=base_outputs)
            print(f'==> sweep λ={lam}', json.dumps(m, ensure_ascii=False))
            append_row({**common, 'ckpt': cand_path,
                        'role': f'sweep_lam_{lam}', 'metrics': m})
            del gen
            torch.cuda.empty_cache()

    print(f'==> 归档: {jsonl}')


if __name__ == '__main__':
    main()
