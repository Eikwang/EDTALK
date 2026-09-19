# -*- coding: utf-8 -*-
"""对比 lip_dim_weight 3.0 vs 5.0 两个 checkpoint 的逐维预测误差（同批 mel）"""
import os, sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import numpy as np
import torch
from networks_audio2lip.audio_encoder import Audio2Lip

CKPT_A = 'logs/Audio2Lip/wy/20260918_0956/checkpoint/epoch_001/066526.pt'  # weight 3.0
CKPT_B = 'logs/Audio2Lip/wy/20260918_1644/checkpoint/epoch_001/066526.pt'  # weight 5.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mA, mB = Audio2Lip().to(device), Audio2Lip().to(device)
mA.load_state_dict(torch.load(CKPT_A, map_location='cpu', weights_only=False)['audio2lip'])
mB.load_state_dict(torch.load(CKPT_B, map_location='cpu', weights_only=False)['audio2lip'])
mA.eval(); mB.eval()

rng = np.random.RandomState(42)
errs_a, errs_b, dyn = [], [], []
with torch.no_grad():
    count = 0
    for mf in sorted(f for f in os.listdir('wy/mel') if f.endswith('.npy')):
        if count >= 4000: break
        mel = np.load(os.path.join('wy/mel', mf)).astype(np.float32)
        lip = np.load(os.path.join('wy/lip_feature', mf)).astype(np.float32)
        n = min(len(mel), len(lip))
        if n < 10: continue
        take = min(n, 200)
        idx = rng.randint(0, max(1, n - take) + 1)
        batch = torch.from_numpy(mel[idx:idx+take]).to(device)
        gt = lip[idx:idx+take]
        pa = mA(batch, 1, len(batch)).reshape(len(batch), 20).cpu().numpy()
        pb = mB(batch, 1, len(batch)).reshape(len(batch), 20).cpu().numpy()
        errs_a.append(np.abs(pa - gt)); errs_b.append(np.abs(pb - gt))
        dyn.append(np.abs(gt))
        count += len(batch)

ea, eb, dyn = np.concatenate(errs_a), np.concatenate(errs_b), np.concatenate(dyn)
print(f'样本数: {count}')
print(f'{"dim":>4} {"err(w3)":>9} {"err(w5)":>9} {"变化":>8} {"GT动态范围":>10}')
for d in range(20):
    tag = ' <-开合' if d in (2, 13, 16) else ''
    print(f'{d:>4} {ea[:,d].mean():>9.4f} {eb[:,d].mean():>9.4f} '
          f'{(eb[:,d].mean()-ea[:,d].mean()):>+8.4f} {np.percentile(dyn[:,d],95):>10.3f}{tag}')
print(f'\n总均值     {ea.mean():>9.4f} {eb.mean():>9.4f} {eb.mean()-ea.mean():>+8.4f}')
o = (2, 13, 16)
print(f'开合维度   {ea[:,o].mean():>9.4f} {eb[:,o].mean():>9.4f} {eb[:,o].mean()-ea[:,o].mean():>+8.4f}')
rest = [d for d in range(20) if d not in o]
print(f'其余维度   {ea[:,rest].mean():>9.4f} {eb[:,rest].mean():>9.4f} {eb[:,rest].mean()-ea[:,rest].mean():>+8.4f}')
