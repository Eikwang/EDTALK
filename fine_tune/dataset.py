import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _sort_frame_files(files):
    """按数字序号排序帧文件，确保 2.png < 10.png < 100.png"""
    def _key(f):
        f = str(f)
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)
    return sorted(files, key=_key)


class Finetune256(Dataset):
    """单人微调数据集：从同一人物视频帧目录采样 source/target 帧对。

    支持两种目录结构:
    1. 扁平目录: video_path/0.png, video_path/1.png, ...
    2. 子目录分组: video_path/sub1/0.png, video_path/sub2/0.png, ...
       (每个子目录视为一个独立视频，时序窗口不会跨越子目录)

    采样策略:
    - 混合采样 (far_ratio > 0):
      以 far_ratio 概率从远距离窗口采样 target (扩展 direction 泛化范围),
      以 1-far_ratio 概率从邻近窗口采样 (保证基础重建质量)。
      训练-推理 OOD 问题的根因: 训练时 source/target 只差 +-30 帧 (约1秒),
      direction 变化小; 推理时 Audio2Lip 产生的 lip direction 幅度远超训练范围,
      Decoder 在大 direction 下行为失控导致模糊。混合采样让 Decoder 见过更大的
      direction 范围。
    - 完全随机采样 (temporal_window 足够大或帧数不足时退化为随机)。
    """

    def __init__(self, video_path, train, transform=None, temporal_window=30,
                 cross_window_min=None, cross_window_max=None,
                 far_window=200, far_ratio=0.3):
        train_test_split = 0.95
        self.train = train

        # 检测目录结构：优先尝试扁平目录，若无则扫描子目录
        direct_frames = _sort_frame_files(
            glob.glob(os.path.join(video_path, '*.png'))
        )

        if len(direct_frames) > 0:
            # 扁平目录结构：所有帧在同一个目录下
            self.video_groups = [direct_frames]
        else:
            # 子目录分组结构：每个子目录视为一个独立视频
            self.video_groups = []
            subdirs = sorted([
                d for d in os.listdir(video_path)
                if os.path.isdir(os.path.join(video_path, d))
            ])
            for subdir in subdirs:
                subdir_path = os.path.join(video_path, subdir)
                frames = _sort_frame_files(
                    glob.glob(os.path.join(subdir_path, '*.png'))
                )
                if len(frames) > 0:
                    self.video_groups.append(frames)

        # 按组内 95/5 划分训练/验证集，确保每个视频都有帧进入训练集和验证集
        self.frames_paths = []
        self.frame_to_group = []   # 每帧所属的视频组索引
        self.group_ranges = []     # 每个视频组在 frames_paths 中的 [start, end) 范围

        current_idx = 0
        for group_idx, frames in enumerate(self.video_groups):
            split_point = int(len(frames) * train_test_split)
            if self.train:
                selected = frames[:split_point]
            else:
                selected = frames[split_point:]

            if len(selected) == 0:
                continue

            start = current_idx
            self.frames_paths.extend(selected)
            current_idx = len(self.frames_paths)
            end = current_idx

            self.group_ranges.append((start, end))
            self.frame_to_group.extend([group_idx] * (end - start))

        self.transform = transform
        # 时序邻近窗口：target 在 [source-window, source+window] 范围内采样
        # 帧数不足时窗口自动收缩到可用范围
        self.temporal_window = max(1, temporal_window)

        # 预计算每个视频组的起止索引，用于快速查找
        self._group_start = [r[0] for r in self.group_ranges]
        self._group_end = [r[1] for r in self.group_ranges]

        # 伪交叉重建采样窗口: cross_target 在 temporal_window 外、cross_window 内采样
        # 目的: 构造"身份不变 + 嘴型来自远距离帧"的伪交叉样本，
        #       提供解耦监督信号 (强制 mask 在背景饱和)
        if cross_window_min is None:
            cross_window_min = temporal_window * 2
        if cross_window_max is None:
            cross_window_max = temporal_window * 4
        self.cross_window_min = max(1, cross_window_min)
        self.cross_window_max = max(self.cross_window_min + 1, cross_window_max)

        # 远距离采样窗口: 以 far_ratio 概率从 [source-far_window, source+far_window]
        # 范围采样 target (排除 temporal_window 内的帧)，扩大 direction 泛化范围
        self.far_window = max(self.temporal_window + 1, far_window)
        self.far_ratio = max(0.0, min(1.0, far_ratio))

    def _get_group_range(self, global_idx):
        """获取 global_idx 所属视频组在 frames_paths 中的 [start, end) 范围"""
        group_idx = self.frame_to_group[global_idx]
        return self.group_ranges[group_idx]

    def __getitem__(self, idx):
        frames_paths = self.frames_paths
        nframes = len(frames_paths)

        # 先确定 source 帧索引（全局随机）
        source_idx = random.randint(0, nframes - 1)

        # 获取 source 所属视频组的范围，时序窗口限制在组内
        group_start, group_end = self._get_group_range(source_idx)
        source_local = source_idx - group_start  # source 在组内的局部索引
        group_size = group_end - group_start

        # ========== target 采样: 混合邻近帧 + 远距离帧 ==========
        # 邻近窗口边界 (temporal_window 内)
        near_lo = max(0, source_local - self.temporal_window)
        near_hi = min(group_size - 1, source_local + self.temporal_window)

        use_far = (self.far_ratio > 0 and random.random() < self.far_ratio)

        if use_far:
            # 远距离采样: 从 [source-far_window, source+far_window] 范围内
            # 排除 temporal_window 内的邻近帧，强制 target 与 source 有较大差异
            far_lo = max(0, source_local - self.far_window)
            far_hi = min(group_size - 1, source_local + self.far_window)
            far_candidates = []
            for c in range(far_lo, far_hi + 1):
                if c < near_lo or c > near_hi:  # 排除邻近帧
                    if c != source_local:         # 不取 source 本身
                        far_candidates.append(c)
            if far_candidates:
                target_local = random.choice(far_candidates)
            else:
                # 降级: 组内随机 (排除 source)
                all_others = [j for j in range(group_size) if j != source_local]
                target_local = random.choice(all_others) if all_others else source_local
        else:
            # 邻近采样: 原始逻辑，保证基础重建质量
            target_candidates = list(range(near_lo, near_hi + 1))
            if source_local in target_candidates and len(target_candidates) > 1:
                target_candidates.remove(source_local)
            target_local = random.choice(target_candidates) if target_candidates else source_local

        # 将局部索引转回全局索引
        target_idx = group_start + target_local

        # ========== cross_target 采样: 在 temporal_window 外、cross_window 内采样 ==========
        # 目的: 构造"身份不变 + 嘴型来自远距离帧"的伪交叉样本
        cross_target_idx = None
        if group_size >= 3:
            cross_lo = max(0, source_local - self.cross_window_max)
            cross_hi = min(group_size - 1, source_local + self.cross_window_max)
            cross_candidates = []
            for c in range(cross_lo, cross_hi + 1):
                if c < near_lo or c > near_hi:  # 不在 temporal_window 范围内
                    if c != source_local:  # 不取 source 本身
                        cross_candidates.append(c)
            if cross_candidates:
                cross_local = random.choice(cross_candidates)
                cross_target_idx = group_start + cross_local
            else:
                # 降级: 完全随机采样 (排除 source)
                all_others = [j for j in range(group_start, group_end) if j != source_idx]
                if all_others:
                    cross_target_idx = random.choice(all_others)

        img_source = Image.open(frames_paths[source_idx]).convert('RGB')
        img_target = Image.open(frames_paths[target_idx]).convert('RGB')

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        # 加载 cross_target
        if cross_target_idx is not None:
            img_cross = Image.open(frames_paths[cross_target_idx]).convert('RGB')
            if self.transform is not None:
                img_cross = self.transform(img_cross)
        else:
            # 无 cross_target: 返回 target 的拷贝作为占位 (trainer 可据此跳过 cross recon)
            img_cross = img_target.clone() if self.transform is not None else img_target

        # NOTE: return 必须在 if 外，否则 transform=None 时会返回 None 导致 DataLoader 崩溃
        return img_source, img_target, img_cross

    def __len__(self):
        return len(self.frames_paths)
