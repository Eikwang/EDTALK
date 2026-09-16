# -*- coding: utf-8 -*-
"""
lip feature 维度语义探测：对单个维度做正向/负向扰动，渲染对比。
确认 dim 16 / 13 / 2 是否为"开合度"相关维度（诊断中误差最集中的维度）。

用法:
  D:\\AI\\EDTalk\\runtime312\\python.exe train/diagnose_lip_dim.py \
      --data_path wy --video 22#14 --frame 9 --dims 16,13,2
"""
import os
import sys
import argparse

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


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def render(gen, identity_img, lip_feats, pose_feats, T):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='wy')
    parser.add_argument('--gen_ckpt', type=str, default='ckpts/EDTalk_lip_pose.pt')
    parser.add_argument('--video', type=str, default='22#14')
    parser.add_argument('--frame', type=int, default=9)
    parser.add_argument('--dims', type=str, default='16,13,2',
                        help='要探测的 lip feature 维度（逗号分隔）')
    parser.add_argument('--delta', type=float, default=1.5, help='扰动幅度')
    parser.add_argument('--out_dir', type=str, default='diagnosis_output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    ckpt = torch.load(args.gen_ckpt, map_location='cpu', weights_only=False)
    gen = Generator(256, 512, 20, 6, 1).to(device)
    gen.load_state_dict(ckpt['gen'] if 'gen' in ckpt else ckpt)
    gen.eval()

    lip = np.load(os.path.join(args.data_path, 'lip_feature', args.video + '.npy'))
    pose = np.load(os.path.join(args.data_path, 'pose_feature', args.video + '.npy'))
    T = 1
    f = args.frame

    lip_t = torch.from_numpy(lip[f:f+T].astype(np.float32)).to(device)
    pose_t = torch.from_numpy(pose[f:f+T].astype(np.float32)).to(device)
    print(f'lip feature @ {args.video} f{f}: {lip_t.cpu().numpy().round(3)}')

    env = lmdb.open(os.path.join(args.data_path, 'lmdb'), readonly=True, lock=False)
    with env.begin(write=False) as txn:
        identity_img = transform(Image.open(
            BytesIO(txn.get(format_for_lmdb(args.video, f))))).to(device)

    dims = [int(d) for d in args.dims.split(',')]
    rows = [lip_t.clone()]  # row 0: 原始
    labels = ['orig']
    for d in dims:
        for sign, tag in ((1.0, '+'), (-1.0, '-')):
            pert = lip_t.clone()
            pert[:, d] += sign * args.delta
            rows.append(pert)
            labels.append(f'dim{d}{tag}')

    all_rows = torch.cat([render(gen, identity_img, r, pose_t, T) for r in rows], dim=0)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'lipdim_{args.video.replace("#", "_")}_f{f}.jpg')
    utils.save_image(all_rows, out, nrow=T, normalize=True, padding=2)
    print(f'[INFO] rows = {labels}')
    print(f'[INFO] saved: {out}')
    env.close()


if __name__ == '__main__':
    main()
