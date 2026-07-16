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
    """单人微调数据集：从同一人物视频帧目录随机采样 source/target 帧对。

    支持两种目录结构:
    1. 扁平目录: video_path/0.png, video_path/1.png, ...
    2. 子目录分组: video_path/sub1/0.png, video_path/sub2/0.png, ...
    """

    def __init__(self, video_path, train, transform=None, **kwargs):
        train_test_split = 0.95
        self.train = train

        # 检测目录结构：优先扁平目录，若无则扫描子目录
        direct_frames = _sort_frame_files(
            glob.glob(os.path.join(video_path, '*.png'))
        )

        if len(direct_frames) > 0:
            all_frames = direct_frames
        else:
            all_frames = []
            subdirs = sorted([
                d for d in os.listdir(video_path)
                if os.path.isdir(os.path.join(video_path, d))
            ])
            for subdir in subdirs:
                subdir_path = os.path.join(video_path, subdir)
                frames = _sort_frame_files(
                    glob.glob(os.path.join(subdir_path, '*.png'))
                )
                all_frames.extend(frames)

        # 训练/验证划分
        split_point = int(len(all_frames) * train_test_split)
        if self.train:
            self.frames_paths = all_frames[:split_point]
        else:
            self.frames_paths = all_frames[split_point:]

        self.transform = transform

    def __getitem__(self, idx):
        frames_paths = self.frames_paths
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.frames_paths)
