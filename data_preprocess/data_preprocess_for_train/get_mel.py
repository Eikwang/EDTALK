# -*- coding: utf-8 -*-
"""
批量提取 Mel 频谱：将目录下所有 wav 音频片段转换为 .npy
用法: python get_mel.py --base_dir HDTF
"""

import os
import sys
import glob
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from demo_EDTalk_A import get_mel


def find_audio_dir(base_dir):
    """自动查找 split_Xs_audio 目录"""
    pattern = os.path.join(base_dir, 'split_*s_audio')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"未找到 split_Xs_audio 目录，请先运行 split_HDTF_audio.py。"
            f"\n  搜索路径: {base_dir}/split_*s_audio"
        )
    return matches[0]


def process_one(audio_path, save_path):
    """处理单个 wav 文件，保存为 .npy"""
    try:
        mel_feature, _, _ = get_mel(audio_path)
        # CUDA tensor -> CPU -> numpy
        mel_feature = mel_feature.cpu().numpy()
        np.save(save_path, mel_feature)
        return True
    except Exception as e:
        print(f"[ERROR] {audio_path}: {e}")
        return False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="数据集主目录（如 HDTF）")

    args = parser.parse_args()

    # 自动查找输入目录：HDTF/split_5s_audio
    audio_dir = find_audio_dir(args.base_dir)

    # 输出目录：HDTF/mel
    mel_dir = os.path.join(args.base_dir, 'mel')
    os.makedirs(mel_dir, exist_ok=True)

    wav_files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')])
    print(f"Input : {audio_dir}")
    print(f"Output: {mel_dir}")
    print(f"Found {len(wav_files)} wav files")

    success_count = 0
    skip_count = 0

    for wav_name in tqdm(wav_files, desc="Extracting mel"):
        stem = wav_name.rsplit('.', 1)[0]
        audio_path = os.path.join(audio_dir, wav_name)
        mel_path = os.path.join(mel_dir, stem + '.npy')

        if os.path.exists(mel_path):
            skip_count += 1
            continue

        if process_one(audio_path, mel_path):
            success_count += 1

    print(f"Done: {success_count} processed, {skip_count} skipped")