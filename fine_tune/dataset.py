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

    采样策略：
    - 时序邻近采样（默认）：source 和 target 取自同一子目录内时间上接近的帧，
      降低姿态/表情差异，让重建任务更可学，减少模型直接复制 target 纹理的倾向。
    - 完全随机采样（temporal_window 足够大或帧数不足时退化为随机）。
    """

    def __init__(self, video_path, train, transform=None, temporal_window=30):
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

        # 按组内 80/20 划分训练/验证集，确保每个视频都有帧进入训练集和验证集
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

        # target 在 source 的时序邻近窗口内采样（限制在组内，不跨视频）
        lo = max(0, source_local - self.temporal_window)
        hi = min(group_size - 1, source_local + self.temporal_window)
        target_candidates = list(range(lo, hi + 1))
        # 确保 target 与 source 不同（帧数 >= 2 时）
        if source_local in target_candidates and len(target_candidates) > 1:
            target_candidates.remove(source_local)
        target_local = random.choice(target_candidates) if target_candidates else source_local

        # 将局部索引转回全局索引
        target_idx = group_start + target_local

        img_source = Image.open(frames_paths[source_idx]).convert('RGB')
        img_target = Image.open(frames_paths[target_idx]).convert('RGB')

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        # NOTE: return 必须在 if 外，否则 transform=None 时会返回 None 导致 DataLoader 崩溃
        return img_source, img_target

    def __len__(self):
        return len(self.frames_paths)
