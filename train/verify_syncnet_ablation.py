# -*- coding: utf-8 -*-
"""
SyncNet 监督消融验证：对比 带 sync_weight=10 (Run A) 与 无 sync (Run B)
两次训练同 iter (066526) 的 audio2lip checkpoint。

输出三层证据:
  1. checkpoint 内 args 的配置差异 (确认唯一变量是 sync)
  2. audio2lip 逐层权重 L2 diff (相对范数)
  3. 同批真实 mel 的 lip feature 预测 diff
     (开合维度 2/13/16 的差 vs 扰动感知阈值 1.5)

用法:
  D:\\AI\\EDTalk\\runtime312\\python.exe train/verify_syncnet_ablation.py
"""
import os
import sys
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

from networks_audio2lip.audio_encoder import Audio2Lip

CKPT_A = 'logs/Audio2Lip/wy/20260917_1729/checkpoint/epoch_001/066526.pt'  # sync_weight=10
CKPT_B = 'logs/Audio2Lip/wy/20260918_0956/checkpoint/epoch_001/066526.pt'  # 无 sync
MEL_DIR = 'wy/mel'


def describe_args(args, tag):
    keep = ['sync_weight', 'syncnet_ckpt', 'epoch', 'lr', 'lr_schedule', 'batch_size',
            'accumulation_steps', 'vgg_sparse_freq', 'vgg_weight', 'smooth_weight',
            'lip_dims', 'lip_dim_weight', 'mouth_vgg_weight', 'audio_encoder_unfreeze_from',
            'resume_ckpt', 'audio2lip_ckpt', 'data_path']
    print(f'\n[{tag}] args:')
    for k in keep:
        v = getattr(args, k, '<未存>')
        print(f'  {k} = {v}')


def weight_diff(sd_a, sd_b):
    print('\n=== audio2lip 逐层权重 diff (相对范数 ||a-b||/||a||) ===')
    overall_a2b, overall_anorm = 0.0, 0.0
    for k in sd_a:
        a, b = sd_a[k].float(), sd_b[k].float()
        d = (a - b).norm().item()
        n = a.norm().item()
        overall_a2b += d * d
        overall_anorm += n * n
        rel = d / max(n, 1e-12)
        if rel > 1e-6:
            print(f'  {k:50s} rel_diff={rel:.6f}  (||a||={n:.2f})')
    rel_total = (overall_a2b ** 0.5) / max(overall_anorm ** 0.5, 1e-12)
    print(f'  总体相对 diff: {rel_total:.8f}  '
          f'(=0 表示权重完全一致; <0.001 表示训练轨迹几乎相同)')
    return rel_total


def lip_predict_diff(ckpt_a, ckpt_b, device, num_windows=2000):
    print('\n=== 同批 mel 的 lip feature 预测 diff ===')
    model_a = Audio2Lip().to(device)
    model_a.load_state_dict(ckpt_a['audio2lip'])
    model_a.eval()
    model_b = Audio2Lip().to(device)
    model_b.load_state_dict(ckpt_b['audio2lip'])
    model_b.eval()

    mel_files = sorted(f for f in os.listdir(MEL_DIR) if f.endswith('.npy'))
    rng = np.random.RandomState(42)
    diffs_all, dims_diff, dims_norm = [], {2: [], 13: [], 16: []}, {2: [], 13: [], 16: []}
    preds_a_all, preds_b_all = [], []

    with torch.no_grad():
        count = 0
        for mf in mel_files:
            if count >= num_windows:
                break
            mel = np.load(os.path.join(MEL_DIR, mf)).astype(np.float32)  # [N,1,80,16]
            n = len(mel)
            take = min(n, 200)
            idx = rng.randint(0, max(1, n - take) + 1)
            batch = torch.from_numpy(mel[idx:idx+take]).to(device)  # [t,1,80,16]
            pa = model_a(batch, 1, len(batch)).reshape(len(batch), 20)
            pb = model_b(batch, 1, len(batch)).reshape(len(batch), 20)
            diffs_all.append((pa - pb).abs().cpu().numpy())
            preds_a_all.append(pa.cpu().numpy())
            preds_b_all.append(pb.cpu().numpy())
            for d in dims_diff:
                dims_norm[d].append(np.abs(pa[:, d].cpu().numpy()))
            count += len(batch)

    diffs_all = np.concatenate(diffs_all)              # [M, 20]
    pa_all = np.concatenate(preds_a_all)
    pb_all = np.concatenate(preds_b_all)

    per_dim = diffs_all.mean(axis=0)
    print(f'  样本窗口数: {count}')
    print(f'  平均绝对 diff (逐维): mean={per_dim.mean():.5f}, '
          f'max=dim{int(per_dim.argmax())}({per_dim.max():.5f})')
    print(f'  全维 diff 的 95 分位: {np.percentile(diffs_all, 95):.5f}')
    print(f'\n  开合度维度对比 (感知阈值: 扰动 ±1.5 肉眼可辨):')
    for d in (2, 13, 16):
        dr = np.abs(pa_all[:, d] - pb_all[:, d])
        print(f'    dim {d}: mean|diff|={dr.mean():.5f}, 95pct={np.percentile(dr, 95):.5f}, '
              f'维度动态范围(95pct)={np.percentile(np.abs(pa_all[:, d]), 95):.3f}')
    # 相对动态范围
    print('\n  判读: 若 mean|diff| << 0.05 且开合维度 diff << 0.1, '
          '则两个模型在功能上不可区分, SyncNet 监督贡献可忽略')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    print(f'loading A (sync): {CKPT_A}')
    ckpt_a = torch.load(CKPT_A, map_location='cpu', weights_only=False)
    print(f'loading B (no sync): {CKPT_B}')
    ckpt_b = torch.load(CKPT_B, map_location='cpu', weights_only=False)

    describe_args(ckpt_a.get('args'), 'Run A (sync_weight=10)')
    describe_args(ckpt_b.get('args'), 'Run B (no sync)')

    sd_a, sd_b = ckpt_a['audio2lip'], ckpt_b['audio2lip']
    print(f'\nstate_dict keys 一致: {list(sd_a.keys()) == list(sd_b.keys())}')
    weight_diff(sd_a, sd_b)
    lip_predict_diff(ckpt_a, ckpt_b, device)


if __name__ == '__main__':
    main()
