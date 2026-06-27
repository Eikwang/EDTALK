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
    """获取图像 LMDB env，单例模式"""
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


# 特征 LMDB 连接缓存 (mel/lip_feature/pose_feature/bbox)
_feat_lmdb_env_cache = {}


def _get_feat_lmdb_env(lmdb_path):
    """获取特征 LMDB env，单例模式 (与图像 LMDB 独立缓存)"""
    if lmdb_path not in _feat_lmdb_env_cache:
        _feat_lmdb_env_cache[lmdb_path] = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
    return _feat_lmdb_env_cache[lmdb_path]



# 样本列表缓存（按数据集路径 + max_samples_per_video 缓存，避免每次启动重新扫描所有 .npy）
_samples_cache = {}


def _get_npy_length(path):
    """只读 .npy header 获取长度，不加载整个数组到内存"""
    return np.load(path, mmap_mode='r').shape[0]


class Audio2LipDataset_image_sync(Dataset):
    def __init__(self, hdtf, is_train=True, transform=None, max_samples_per_video=None,
                 preload_features=False):
        self.is_train = is_train
        self.hdtf_path = hdtf
        self.transform = transform
        self.max_samples_per_video = max_samples_per_video

        # 特征 LMDB 路径: {base_dir}/lmdb_features (可选, 不存在则回退 .npy)
        feat_lmdb_path = os.path.join(hdtf, 'lmdb_features')
        self.feat_env = None
        self._feature_cache = None  # preload_features=True 时为 dict, 否则 None
        if os.path.exists(feat_lmdb_path):
            self.feat_env = _get_feat_lmdb_env(feat_lmdb_path)
            print(f"[Audio2LipDataset] 检测到特征 LMDB: {feat_lmdb_path}")
        else:
            print(f"[Audio2LipDataset] 未找到特征 LMDB ({feat_lmdb_path}), "
                  f"将回退到 .npy 文件读取 (建议运行 prepare_feature_lmdb.py 预处理)")

        # 图像 LMDB 路径：{base_dir}/lmdb，使用缓存避免重复打开
        lmdb_path = os.path.join(hdtf, 'lmdb')
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
        self.env = _get_lmdb_env(lmdb_path)

        # 从已打开的 env 中提取视频列表（避免重复打开同一 LMDB 路径）
        self.video_list = _extract_video_names_from_env(self.env)

        # 缓存 key：数据集路径 + 采样参数，train/test 共享同一份 samples 列表
        cache_key = (os.path.abspath(hdtf), max_samples_per_video)

        if cache_key not in _samples_cache:
            # 首次构建：扫描所有视频，预计算 (video_idx, start_frame) 列表
            print(f"[Audio2LipDataset] 首次扫描视频列表（缓存后续启动将秒加载）...")
            t0 = time.time()
            samples = []
            # 批量读取 LMDB length（单次事务遍历所有视频）
            with self.env.begin(write=False) as txn:
                for v_idx, name in enumerate(self.video_list):
                    audio_path = os.path.join(hdtf, 'mel', name + '.npy')
                    lip_path = os.path.join(hdtf, 'lip_feature', name + '.npy')
                    pose_path = os.path.join(hdtf, 'pose_feature', name + '.npy')

                    if not all(os.path.exists(p) for p in [audio_path, lip_path, pose_path]):
                        continue

                    # mmap_mode='r' 只读 header 获取 shape[0]，不加载整个数组
                    audio_len = _get_npy_length(audio_path)
                    lip_len = _get_npy_length(lip_path)
                    pose_len = _get_npy_length(pose_path)

                    key = format_for_lmdb(name, 'length')
                    length = int(txn.get(key).decode('utf-8'))

                    l = min(audio_len, lip_len, pose_len, length)
                    max_start = l - 5  # 需要连续5帧: [r, r+5)
                    if max_start <= 0:
                        continue

                    if max_samples_per_video is not None and max_samples_per_video < max_start:
                        step = max(1, max_start // max_samples_per_video)
                        starts = list(range(0, max_start, step))[:max_samples_per_video]
                    else:
                        starts = list(range(max_start))

                    for start in starts:
                        samples.append((v_idx, start))

            _samples_cache[cache_key] = samples
            print(f"[Audio2LipDataset] 扫描完成: {len(self.video_list)} videos, "
                  f"{len(samples)} clips, 耗时 {time.time()-t0:.1f}s")

        # train/test 共享同一份 samples 列表（拷贝避免 shuffle 互相影响）
        self.samples = list(_samples_cache[cache_key])
        print(f"[Audio2LipDataset] Videos: {len(self.video_list)}, Total clips: {len(self.samples)}")
        if is_train:
            random.shuffle(self.samples)

        # 可选: 将所有特征预加载到内存 (适用于小数据集微调, 消除训练中所有特征 IO)
        if preload_features:
            self._preload_all_features()

    def _load_feature(self, feature_name, video_name):
        """从特征 LMDB / 内存缓存 / .npy 文件加载特征数组 (三级回退)

        优先级:
          1. 内存缓存 (preload_features=True 时)
          2. 特征 LMDB (lmdb_features 目录存在时)
          3. .npy 文件 (回退, 保持兼容)

        Args:
            feature_name: 'mel' / 'lip_feature' / 'pose_feature' / 'bbox'
            video_name: 视频名 (不含扩展名)

        Returns:
            numpy 数组 (整个视频的特征序列)
        """
        cache_key = f"{feature_name}-{video_name}"

        # 1. 内存缓存
        if self._feature_cache is not None:
            arr = self._feature_cache.get(cache_key)
            if arr is not None:
                return arr

        # 2. 特征 LMDB
        if self.feat_env is not None:
            with self.feat_env.begin(write=False) as txn:
                data = txn.get(cache_key.encode('utf-8'))
            if data is not None:
                return np.load(BytesIO(data))

        # 3. 回退到 .npy 文件
        npy_path = os.path.join(self.hdtf_path, feature_name, video_name + '.npy')
        return np.load(npy_path)

    def _preload_all_features(self):
        """将所有视频的 mel/lip_feature/pose_feature/bbox 预加载到内存

        适用于小数据集微调 (如单人视频): 一次性加载后 __getitem__ 零 IO。
        大数据集不建议使用 (内存消耗大)。

        数据来源优先级: 特征 LMDB > .npy 文件
        """
        feature_names = ['mel', 'lip_feature', 'pose_feature', 'bbox']
        self._feature_cache = {}
        total_bytes = 0

        if self.feat_env is not None:
            # 从特征 LMDB 批量加载
            with self.feat_env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode('utf-8')
                    if key_str.startswith('__'):
                        continue  # 跳过元数据键
                    arr = np.load(BytesIO(value))
                    self._feature_cache[key_str] = arr
                    total_bytes += arr.nbytes
        else:
            # 从 .npy 文件批量加载
            for name in self.video_list:
                for feat in feature_names:
                    npy_path = os.path.join(self.hdtf_path, feat, name + '.npy')
                    if os.path.exists(npy_path):
                        arr = np.load(npy_path)
                        cache_key = f"{feat}-{name}"
                        self._feature_cache[cache_key] = arr
                        total_bytes += arr.nbytes

        total_mb = total_bytes / 1024 / 1024
        print(f"[Audio2LipDataset] 预加载 {len(self._feature_cache)} 个特征到内存, "
              f"总计 {total_mb:.1f} MB")

    # -- DataLoader 多进程兼容 --
    # Windows spawn 模式下 worker 进程需要 pickle dataset,
    # 但 lmdb.Environment 不可 pickle。这里在序列化时移除 env 引用,
    # 在 worker 进程反序列化时通过模块级缓存重新打开。

    def __getstate__(self):
        """pickle 时移除不可序列化的 LMDB env 对象"""
        state = self.__dict__.copy()
        state['env'] = None
        state['feat_env'] = None
        return state

    def __setstate__(self, state):
        """反序列化时恢复状态并重新打开 LMDB env"""
        self.__dict__.update(state)
        # 重新打开图像 LMDB (必需, __getitem__ 要读图像)
        lmdb_path = os.path.join(self.hdtf_path, 'lmdb')
        if os.path.exists(lmdb_path):
            self.env = _get_lmdb_env(lmdb_path)
        # 重新打开特征 LMDB (仅在未预加载到内存时需要)
        if self._feature_cache is None:
            feat_lmdb_path = os.path.join(self.hdtf_path, 'lmdb_features')
            if os.path.exists(feat_lmdb_path):
                self.feat_env = _get_feat_lmdb_env(feat_lmdb_path)

    def __len__(self):
        # 关键修改：返回总片段数，而非视频数
        # 这样每个 epoch DataLoader 会遍历所有 (视频, 起始帧) 组合
        return len(self.samples)

    def __getitem__(self, idx):
        # 关键修改：根据全局 idx 直接定位到 (视频索引, 起始帧)
        v_idx, r = self.samples[idx]
        name_a = self.video_list[v_idx]
        

        # 从特征 LMDB / 内存缓存 / .npy 加载 (三级回退, 消除重复磁盘 IO)
        audio_features = self._load_feature('mel', name_a)
        lip_features = self._load_feature('lip_feature', name_a)
        pose_features = self._load_feature('pose_feature', name_a)
        bbox_features = self._load_feature('bbox', name_a)

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
