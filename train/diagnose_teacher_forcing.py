# -*- coding: utf-8 -*-
"""
Teacher Forcing 诊断：定位爆破音不闭合的病灶。

三层对比（每个片段一行）：
  GT        : 数据集真实帧（ground truth）
  Teacher   : 真实 lip_features + 真实 pose_features -> 底模 Generator
  Predicted : mel -> 微调 audio2lip -> 预测 lip_features + 真实 pose -> Generator

判读:
  Teacher 闭口正确 + Predicted 不闭合 -> 病灶在 audio2lip 预测（回归均值），走数据过采样/闭合加权
  Teacher 也不闭合/牙齿不像          -> 病灶在 Generator dec 先验，走 fine_tune 嘴区加权重心调整

片段筛选（三类）:
  closure : mel 能量低谷（疑似闭口/静音段）
  action  : lip_feature 帧间差分大（唇部动作密集）
  random  : 随机对照

用法:
  D:\\AI\\EDTalk\\runtime312\\python.exe train/diagnose_teacher_forcing.py \
      --data_path wy --audio2lip_ckpt logs/Audio2Lip/wy/<date>/checkpoint/epoch_002/047289.pt
"""
import os
import sys
import argparse
import random

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torchvision
from PIL import Image
from io import BytesIO
import lmdb
from torchvision import utils

from networks_Lip_NonLip.generator import Generator
from networks_audio2lip.audio_encoder import Audio2Lip

CLIP_T = 5  # 与训练一致: 每片段 5 帧


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def load_generator(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['gen'] if 'gen' in ckpt else ckpt
    gen = Generator(256, 512, 20, 6, 1).to(device)
    gen.load_state_dict(state)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False
    print(f'[INFO] Generator loaded from {ckpt_path}')
    return gen


def load_audio2lip(ckpt_path, device):
    model = Audio2Lip().to(device)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt['audio2lip'] if 'audio2lip' in ckpt else ckpt
        model.load_state_dict(state)
        src = ckpt_path
    else:
        src = 'Wav2Lip init (Audio2Lip.__init__ 默认加载)'
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f'[INFO] Audio2Lip loaded from {src}')
    return model


def render(gen, identity_img, lip_feats, pose_feats, T):
    """lip/pose -> Generator 渲染。identity_img: [3,256,256], lip/pose: [T, dim]"""
    with torch.no_grad():
        img = identity_img.unsqueeze(0)
        wa_identity, _, feats_identity, _ = gen.enc(img)
        alpha = torch.cat([lip_feats, pose_feats], dim=-1)
        directions = gen.direction_lipnonlip(alpha)
        rep = torch.LongTensor([T]).to(lip_feats.device)
        wa_rep = torch.repeat_interleave(wa_identity, rep, dim=0)
        feats_rep = [torch.repeat_interleave(f, rep, dim=0) for f in feats_identity]
        latent = wa_rep + directions
        recon = gen.dec(latent, None, feats_rep)
        return recon.reshape(T, 3, 256, 256)


def predict_lip(audio2lip, mel_windows, T):
    """mel 窗口 [T, 80, 16]（.npy 可能已带通道维 [T, 1, 80, 16]）-> 预测 lip feature [T, 20]"""
    with torch.no_grad():
        x = mel_windows
        if x.dim() == 4:      # [T, 1, 80, 16] -> [T, 80, 16]
            x = x.squeeze(1)
        x = x.unsqueeze(1)    # [T, 1, 80, 16]
        out = audio2lip(x, 1, T)      # [1, T, 20]
        return out.reshape(T, 20)


def select_clips(data_path, num_per_type, max_frames=4000):
    """按三类筛选片段: 返回 [(video, start_frame, type), ...]"""
    mel_dir = os.path.join(data_path, 'mel')
    lip_dir = os.path.join(data_path, 'lip_feature')
    videos = sorted(
        f.rsplit('.', 1)[0]
        for f in os.listdir(mel_dir)
        if f.endswith('.npy')
    )
    random.seed(42)
    picks = {'closure': [], 'action': [], 'random': []}

    for v in videos:
        mel = np.load(os.path.join(mel_dir, v + '.npy'))     # [N, 80, 16]
        lip = np.load(os.path.join(lip_dir, v + '.npy'))     # [N, 20]
        n = min(len(mel), len(lip)) - CLIP_T
        if n < CLIP_T * 2:
            continue

        # mel 能量: 每帧窗口的均值（低 = 疑似静音/闭合）
        energy = mel[:n].reshape(n, -1).mean(axis=1)
        # 动作强度: lip feature 帧间差分
        motion = np.linalg.norm(np.diff(lip[:n], axis=0), axis=1)

        # closure: 能量最低的片段（排除全静音视频开头）
        lo = CLIP_T
        e_thresh = np.percentile(energy[lo:], 5)
        cand = [i for i in range(lo, n - 1) if energy[i] <= e_thresh]
        random.shuffle(cand)
        for i in cand[:max(1, num_per_type // max(1, len(videos)))]:
            picks['closure'].append((v, i))

        # action: 动作最剧烈的片段 (motion[i] 为帧 i->i+1 的差分, 有效到 n-2)
        a_thresh = np.percentile(motion[lo:], 95)
        cand = [i for i in range(lo, n - 1) if motion[i] >= a_thresh]
        random.shuffle(cand)
        for i in cand[:max(1, num_per_type // max(1, len(videos)))]:
            picks['action'].append((v, i))

        # random: 对照
        for _ in range(max(1, num_per_type // (5 * max(1, len(videos))))):
            picks['random'].append((v, random.randint(lo, n - 1)))

    out = []
    for typ in ('closure', 'action', 'random'):
        random.shuffle(picks[typ])
        out.extend((v, i, typ) for v, i in picks[typ][:num_per_type])
    return out


def main():
    parser = argparse.ArgumentParser(description='Teacher Forcing 诊断')
    parser.add_argument('--data_path', type=str, default='wy')
    parser.add_argument('--gen_ckpt', type=str, default='ckpts/EDTalk_lip_pose.pt')
    parser.add_argument('--audio2lip_ckpt', type=str, default=None,
                        help='微调后的 audio2lip checkpoint（含 audio2lip 键）。不传则 Predicted 行用 Wav2Lip 初始化')
    parser.add_argument('--num_clips', type=int, default=9, help='每类片段数')
    parser.add_argument('--out_dir', type=str, default='diagnosis_output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    gen = load_generator(args.gen_ckpt, device)
    audio2lip = load_audio2lip(args.audio2lip_ckpt, device)

    # 打开图像 LMDB
    lmdb_path = os.path.join(args.data_path, 'lmdb')
    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                    readahead=False, meminit=False)

    os.makedirs(args.out_dir, exist_ok=True)
    clips = select_clips(args.data_path, args.num_clips)
    print(f'[INFO] selected {len(clips)} clips '
          f'(closure/action/random x {args.num_clips})')

    report_lines = []
    for ci, (video, start, typ) in enumerate(clips):
        # 加载特征
        mel = np.load(os.path.join(args.data_path, 'mel', video + '.npy'))
        lip = np.load(os.path.join(args.data_path, 'lip_feature', video + '.npy'))
        pose = np.load(os.path.join(args.data_path, 'pose_feature', video + '.npy'))

        with env.begin(write=False) as txn:
            length = int(txn.get(format_for_lmdb(video, 'length')).decode('utf-8'))
        n = min(min(len(mel), len(lip), len(pose)), length)

        if start + CLIP_T > n:
            start = n - CLIP_T

        mel_clip = torch.from_numpy(mel[start:start+CLIP_T].astype(np.float32)).to(device)
        lip_clip = torch.from_numpy(lip[start:start+CLIP_T].astype(np.float32)).to(device)
        pose_clip = torch.from_numpy(pose[start:start+CLIP_T].astype(np.float32)).to(device)

        # identity 帧: 片段第 0 帧（与训练一致从同视频取）
        with env.begin(write=False) as txn:
            key = format_for_lmdb(video, start)
            img_bytes = txn.get(key)
            identity_img = transform(Image.open(BytesIO(img_bytes))).to(device)

        # 真实帧 GT
        gt_frames = []
        with env.begin(write=False) as txn:
            for fi in range(start, start + CLIP_T):
                img_bytes = txn.get(format_for_lmdb(video, fi))
                gt_frames.append(transform(Image.open(BytesIO(img_bytes))))
        gt = torch.stack(gt_frames).to(device)  # [T, 3, 256, 256]

        # Teacher forcing: 真实 lip + 真实 pose
        recon_teacher = render(gen, identity_img, lip_clip, pose_clip, CLIP_T)

        # Predicted: mel -> audio2lip -> 预测 lip
        lip_pred = predict_lip(audio2lip, mel_clip, CLIP_T)
        recon_pred = render(gen, identity_img, lip_pred, pose_clip, CLIP_T)

        # 三行: GT / Teacher / Predicted
        rows = torch.cat([gt, recon_teacher.clamp(-1, 1), recon_pred.clamp(-1, 1)], dim=0)
        out_path = os.path.join(
            args.out_dir, f'{typ}_{ci:03d}_{video}_f{start}.jpg')
        utils.save_image(rows, out_path, nrow=CLIP_T, normalize=True, padding=2)

        # 数值对比: 预测 lip vs 真实 lip 的逐维误差
        lip_err = (lip_pred - lip_clip).abs().mean(dim=0)  # [20]
        report_lines.append(
            f'{typ} {video} f{start}: lip_err mean={lip_err.mean():.4f} '
            f'max_dim={int(lip_err.argmax())}({lip_err.max():.4f})')
        print(report_lines[-1])

    with open(os.path.join(args.out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    env.close()
    print(f'\n[INFO] done. 对比图与 report.txt 已保存到 {args.out_dir}/')
    print('判读: Teacher 行闭口正确但 Predicted 行不闭合 -> audio2lip 病灶;'
          ' Teacher 行也不闭合 -> Generator dec 先验病灶')


if __name__ == '__main__':
    main()
