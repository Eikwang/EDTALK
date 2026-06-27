import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Finetune256(Dataset):
    """单人微调数据集：从同一人物视频帧目录采样 source/target 帧对。

    采样策略：
    - 时序邻近采样（默认）：source 和 target 取自时间上接近的帧，
      降低姿态/表情差异，让重建任务更可学，减少模型直接复制 target 纹理的倾向。
    - 完全随机采样（temporal_window 足够大或帧数不足时退化为随机）。
    """

    def __init__(self, video_path, train, transform=None, temporal_window=30):
        train_test_split = 0.8
        self.train = train

        # 优先搜索当前目录下的 png，若无则递归搜索子目录
        frames_paths = sorted(glob.glob(os.path.join(video_path, '*.png')))
        if len(frames_paths) == 0:
            frames_paths = sorted(glob.glob(os.path.join(video_path, '**', '*.png'), recursive=True))
        video_len = int(len(frames_paths) * train_test_split)

        if self.train:
            self.frames_paths = frames_paths[:video_len]
        else:
            self.frames_paths = frames_paths[video_len:]
        self.transform = transform
        # 时序邻近窗口：target 在 [source-window, source+window] 范围内采样
        # 帧数不足时窗口自动收缩到可用范围
        self.temporal_window = max(1, temporal_window)

    def __getitem__(self, idx):
        frames_paths = self.frames_paths
        nframes = len(frames_paths)

        # 先确定 source 帧索引
        source_idx = random.randint(0, nframes - 1)
        # target 在 source 的时序邻近窗口内采样（边界处做 clamp）
        lo = max(0, source_idx - self.temporal_window)
        hi = min(nframes - 1, source_idx + self.temporal_window)
        target_candidates = list(range(lo, hi + 1))
        # 确保 target 与 source 不同（帧数 >= 2 时）
        if source_idx in target_candidates and len(target_candidates) > 1:
            target_candidates.remove(source_idx)
        target_idx = random.choice(target_candidates) if target_candidates else source_idx

        img_source = Image.open(frames_paths[source_idx]).convert('RGB')
        img_target = Image.open(frames_paths[target_idx]).convert('RGB')

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        # NOTE: return 必须在 if 外，否则 transform=None 时会返回 None 导致 DataLoader 崩溃
        return img_source, img_target

    def __len__(self):
        return len(self.frames_paths)
