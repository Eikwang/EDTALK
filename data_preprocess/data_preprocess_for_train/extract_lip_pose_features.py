# -*- coding: utf-8 -*-
"""
从 LMDB 中提取每帧的 lip 特征 (20维) 和 pose 特征 (6维)
用法: python extract_lip_pose_features.py --base_dir HDTF
"""

import os
import sys
import lmdb
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
import argparse
from tqdm import tqdm

# ============================================================
# 项目路径配置
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

# 使用推理版 Generator（已添加 get_lip_pose_feature 方法）
# 配合 EDTalk_lip_pose.pt 使用
from networks.generator_lip_pose import Generator


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def get_video_list_from_lmdb(lmdb_path):
    """从 LMDB 中读取所有视频名列表"""
    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        raise IOError(f'Cannot open lmdb dataset: {lmdb_path}')

    videos = set()
    with env.begin(write=False) as txn:
        length_key = format_for_lmdb('length')
        total_bytes = txn.get(length_key)
        total = int(total_bytes.decode('utf-8')) if total_bytes else 0

        cursor = txn.cursor()
        for key, _ in cursor:
            key_str = key.decode('utf-8')
            if '-length' in key_str:
                video_name = key_str.rsplit('-length', 1)[0]
            else:
                parts = key_str.rsplit('-', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    video_name = parts[0]
                else:
                    continue
            if video_name:
                videos.add(video_name)

    env.close()
    result = sorted(videos)

    if len(result) == 0 and total > 0:
        print(f"[WARN] 无法从 LMDB key 解析视频名，尝试按索引查找 (total={total})")
        return [str(i) for i in range(total)]

    return result


class LipPoseFeatureExtractor:
    """使用预训练 Generator 提取 lip 和 pose 特征"""

    def __init__(self, ckpt_path, device='cuda'):
        self.device = device

        print(f'==> Loading Generator from {ckpt_path}')
        self.generator = Generator().to(device)

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
        self.generator.load_state_dict(checkpoint['gen'])
        self.generator.eval()

        for param in self.generator.parameters():
            param.requires_grad = False

        print('==> Generator loaded successfully')

    def extract_features(self, img_tensor):
        """
        从单张图像提取特征

        Args:
            img_tensor: [1, 3, 256, 256] tensor, normalized to [-1, 1]

        Returns:
            lip_feature: numpy array (20,)
            pose_feature: numpy array (6,)
        """
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            lip_feat, pose_feat = self.generator.get_lip_pose_feature(img_tensor)
            return lip_feat.squeeze(0).cpu().numpy(), pose_feat.squeeze(0).cpu().numpy()


def process_video(video_name, lmdb_path, extractor, lip_save_dir, pose_save_dir):
    """处理单个视频：提取所有帧的 lip/pose 特征并保存"""
    lip_save_path = os.path.join(lip_save_dir, video_name + '.npy')
    pose_save_path = os.path.join(pose_save_dir, video_name + '.npy')

    # 跳过已处理（两个文件都存在才跳过）
    if os.path.exists(lip_save_path) and os.path.exists(pose_save_path):
        return True, video_name

    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

    try:
        with env.begin(write=False) as txn:
            key = format_for_lmdb(video_name, 'length')
            length_bytes = txn.get(key)
            if length_bytes is None:
                return False, video_name

            length = int(length_bytes.decode('utf-8'))

            lip_features = []
            pose_features = []

            for frame_idx in range(length):
                key = format_for_lmdb(video_name, frame_idx)
                img_bytes = txn.get(key)

                if img_bytes is None:
                    continue

                # 加载并预处理图像
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                img = img.resize((256, 256))
                img = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
                img = img.astype(np.float32) / 255.0       # [0, 1]
                img = torch.from_numpy(img).unsqueeze(0)   # [1, 3, 256, 256]
                img = (img - 0.5) * 2.0                   # [-1, 1]

                # 提取特征
                lip_feat, pose_feat = extractor.extract_features(img)
                lip_features.append(lip_feat)
                pose_features.append(pose_feat)

        if len(lip_features) == 0:
            return False, video_name

        # 转换为 numpy 并保存
        lip_array = np.array(lip_features)   # [num_frames, 20]
        pose_array = np.array(pose_features)  # [num_frames, 6]

        np.save(lip_save_path, lip_array)
        np.save(pose_save_path, pose_array)

        return True, video_name

    except Exception as e:
        print(f"[ERROR] {video_name}: {e}")
        return False, video_name
    finally:
        env.close()


def find_lmdb_path(base_dir):
    """自动查找 LMDB 目录"""
    lmdb_path = os.path.join(base_dir, 'lmdb')
    data_mdb = os.path.join(lmdb_path, 'data.mdb')
    if not os.path.exists(data_mdb):
        raise FileNotFoundError(
            f"未找到 LMDB 数据库，请先运行 prepare_lmdb.py。\n"
            f"期望路径: {data_mdb}"
        )
    return lmdb_path


def find_ckpt_path():
    """查找 EDTalk 预训练模型
    
    注意：必须使用 EDTalk_lip_pose.pt，不能使用 EDTalk.pt！
    因为 EDTalk.pt 包含 expression 模块（direction_exp, exp_fc, fineadainresblock），
    与 generator_lip_pose.py 的结构不兼容。
    """
    candidates = [
        os.path.join(_PROJECT_ROOT, 'ckpts', 'EDTalk_lip_pose.pt'),
        'EDTalk_lip_pose.pt',
        os.path.join(_PROJECT_ROOT, 'ckpts', 'EDTalk_lip_pose.pt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "未找到 EDTalk_lip_pose.pt 模型。\n"
        f"请确保模型文件存在于以下位置之一: {candidates}\n"
        "注意：必须使用 EDTalk_lip_pose.pt，不能使用 EDTalk.pt"
    )


def main(base_dir, gpu_id=0, force=False):
    """
    从 LMDB 提取每帧的 lip 和 pose 特征

    用法: python extract_lip_pose_features.py --base_dir HDTF [--gpu_id 0]
    """

    print('=' * 60)
    print('EDTalk Lip/Pose Feature Extraction')
    print('=' * 60)

    # 设备选择
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f'==> Using device: {device}')

    # 自动查找路径
    lmdb_path = find_lmdb_path(base_dir)
    ckpt_path = find_ckpt_path()

    # 输出目录：{base_dir}/lip_feature 和 {base_dir}/pose_feature
    lip_save_dir = os.path.join(base_dir, 'lip_feature')
    pose_save_dir = os.path.join(base_dir, 'pose_feature')
    os.makedirs(lip_save_dir, exist_ok=True)
    os.makedirs(pose_save_dir, exist_ok=True)

    print(f'==> LMDB     : {lmdb_path}')
    print(f'==> Model    : {ckpt_path}')
    print(f'==> Lip out  : {lip_save_dir}')
    print(f'==> Pose out : {pose_save_dir}')

    # 初始化特征提取器
    extractor = LipPoseFeatureExtractor(ckpt_path, device=device)

    # 从 LMDB 读取视频列表
    videos = get_video_list_from_lmdb(lmdb_path)
    print(f'==> Found {len(videos)} videos')

    # 过滤已处理的
    pending = []
    for v in videos:
        lip_p = os.path.join(lip_save_dir, v + '.npy')
        pose_p = os.path.join(pose_save_dir, v + '.npy')
        if force or not (os.path.exists(lip_p) and os.path.exists(pose_p)):
            pending.append(v)

    if len(pending) < len(videos):
        print(f'==> Skip {len(videos) - len(pending)} already processed')

    if not pending:
        print('==> All done!')
        return

    success = 0
    fail = 0

    # 逐个处理（GPU 推理，串行更稳定）
    for vfile in tqdm(pending, desc='Extracting features'):
        try:
            ok, _ = process_video(vfile, lmdb_path, extractor, lip_save_dir, pose_save_dir)
            if ok:
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f'[ERROR] {vfile}: {e}')
            fail += 1

    print(f'\n==> Done! {success} processed, {fail} failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从 LMDB 提取 lip 和 pose 特征')
    parser.add_argument('--base_dir', type=str, required=True, help='数据集主目录（如 HDTF）')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--force', action='store_true', help='强制重新处理已有文件')
    args = parser.parse_args()

    main(args.base_dir, args.gpu_id, args.force)
