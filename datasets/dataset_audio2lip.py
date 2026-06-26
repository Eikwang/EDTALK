import os
from unittest import main
from skimage import io, img_as_float32, transform
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import glob
import pickle
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch, json
from io import BytesIO
import cv2
import torch.nn.functional as F
from torchvision import utils
import numpy as np
from PIL import Image
import cv2
import skimage.transform as trans
import lmdb
import torchvision

    

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def _extract_video_names_from_env(env):
    """从已打开的 LMDB env 中提取所有视频名列表"""
    videos = set()
    with env.begin(write=False) as txn:
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
    return sorted(videos)


# LMDB 连接缓存（避免同一进程多次打开同一路径）
_lmdb_env_cache = {}


def _get_lmdb_env(lmdb_path):
    """获取 LMDB env，单例模式"""
    if lmdb_path not in _lmdb_env_cache:
        _lmdb_env_cache[lmdb_path] = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
    return _lmdb_env_cache[lmdb_path]



class Audio2LipDataset_image_sync(Dataset):
    def __init__(self, hdtf, is_train=True, transform=None, max_samples_per_video=None):
        self.is_train = is_train
        self.hdtf_path = hdtf
        self.transform = transform
        self.max_samples_per_video = max_samples_per_video

        # LMDB 路径：{base_dir}/lmdb，使用缓存避免重复打开
        lmdb_path = os.path.join(hdtf, 'lmdb')
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
        self.env = _get_lmdb_env(lmdb_path)

        # 从已打开的 env 中提取视频列表（避免重复打开同一 LMDB 路径）
        self.video_list = _extract_video_names_from_env(self.env)

        # 预计算所有可采样片段: (video_idx, start_frame)
        # 关键修改：按帧采样而非按视频采样，确保每个 epoch 能遍历全部数据
        self.samples = []
        for v_idx, name in enumerate(self.video_list):
            # 获取各特征长度
            audio_path = os.path.join(hdtf, 'mel', name + '.npy')
            lip_path = os.path.join(hdtf, 'lip_feature', name + '.npy')
            pose_path = os.path.join(hdtf, 'pose_feature', name + '.npy')
            
            if not all(os.path.exists(p) for p in [audio_path, lip_path, pose_path]):
                continue
            
            audio_len = len(np.load(audio_path))
            lip_len = len(np.load(lip_path))
            pose_len = len(np.load(pose_path))
            
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(name, 'length')
                length = int(txn.get(key).decode('utf-8'))
            
            l = min(audio_len, lip_len, pose_len, length)
            max_start = l - 5  # 需要连续5帧: [r, r+5)
            if max_start <= 0:
                continue
            
            # 限制每个视频的采样数（可选，用于控制 epoch 大小）
            # 设置为 None 则使用所有可能帧
            if max_samples_per_video is not None and max_samples_per_video < max_start:
                step = max(1, max_start // max_samples_per_video)
                starts = list(range(0, max_start, step))[:max_samples_per_video]
            else:
                starts = list(range(max_start))
            
            for start in starts:
                self.samples.append((v_idx, start))
        
        print(f"[Audio2LipDataset] Videos: {len(self.video_list)}, Total clips: {len(self.samples)}")
        if is_train:
            random.shuffle(self.samples)

    def __len__(self):
        # 关键修改：返回总片段数，而非视频数
        # 这样每个 epoch DataLoader 会遍历所有 (视频, 起始帧) 组合
        return len(self.samples)

    def __getitem__(self, idx):
        # 关键修改：根据全局 idx 直接定位到 (视频索引, 起始帧)
        v_idx, r = self.samples[idx]
        name_a = self.video_list[v_idx]
        

        audio_path = os.path.join(self.hdtf_path, 'mel', name_a + '.npy')
        audio_features = np.load(audio_path)

        lip_path = os.path.join(self.hdtf_path, 'lip_feature', name_a + '.npy')
        lip_features = np.load(lip_path)

        pose_path = os.path.join(self.hdtf_path, 'pose_feature', name_a + '.npy')
        pose_features = np.load(pose_path)

        bbox_path = os.path.join(self.hdtf_path, 'bbox', name_a + '.npy')
        bbox_features = np.load(bbox_path)

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(name_a, 'length')
            length = int(txn.get(key).decode('utf-8'))
        l = min(min(len(audio_features), len(lip_features)),length)
        
        # r_identity: 使用确定性伪随机，基于全局 idx
        # 确保同一 idx 总是返回相同的 identity 帧（便于复现），同时保持合理随机性
        r_identity = (idx * 31 + 17) % max(1, l - 1)
        


        image_list = []

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(name_a, r_identity)
            img_bytes = txn.get(key)
            identity_img = self.transform(Image.open(BytesIO(img_bytes)))
            for current_frame in range(r,r+5):
                key = format_for_lmdb(name_a, current_frame)
                img_bytes = txn.get(key)
                image_list.append(self.transform(Image.open(BytesIO(img_bytes))))

        image_list = torch.stack(image_list, dim=0)

        data = {}
        data['audio_features'] = audio_features[r:r+5]
        data['lip_features'] = lip_features[r:r+5]
        data['pose_features'] = pose_features[r:r+5]
        data['identity_img'] = identity_img
        data['target_img'] = image_list

        bbox_len = len(bbox_features)
        if r+5 <= bbox_len:
            data['bbox'] = bbox_features[r:r+5]
        else:
            bbox = []

            for i in range(r,r+5):
                try:
                    temp = bbox_features[i]
                except:
                    temp = bbox_features[-1]
                bbox.append(temp)
            bbox = np.array(bbox)
            data['bbox'] = bbox
        return data
