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
    """微调数据集：从视频中随机采样 source/target 帧对。

    采样策略 (same_identity_ratio 控制):
    - 同身份 (默认80%): 从同一视频中选2帧，source和target是同一人的不同姿态/口型
      → 匹配推理分布 (推理时source和driving是同一人)
    - 跨身份 (20%): 从不同视频各选1帧，保持一定的泛化能力

    支持两种目录结构:
    1. 扁平目录: video_path/0.jpg, video_path/1.jpg, ...
    2. 子目录分组: video_path/sub1/0.jpg, video_path/sub2/0.jpg, ...

    支持的图像格式: jpg, jpeg, png, bmp
    """

    def __init__(self, video_path, transform=None, same_identity_ratio=0.8, **kwargs):
        # 支持的图像扩展名（按优先级：jpg 解码更快，png 无损）
        img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

        def _collect_frames(path):
            """收集目录下的所有图像帧（jpg/jpeg/png/bmp）"""
            files = []
            for ext in img_exts:
                files.extend(glob.glob(os.path.join(path, ext)))
            return _sort_frame_files(files)

        # 检测目录结构：按视频分组存储
        self.video_frames = []  # list of lists: [[video1_frames], [video2_frames], ...]
        self.video_names = []

        direct_frames = _collect_frames(video_path)

        if len(direct_frames) > 0:
            # 扁平目录：整个目录视为一个视频
            self.video_frames.append(direct_frames)
            self.video_names.append(os.path.basename(video_path))
        else:
            # 子目录分组：每个子目录是一个视频
            subdirs = sorted([
                d for d in os.listdir(video_path)
                if os.path.isdir(os.path.join(video_path, d))
            ])
            for subdir in subdirs:
                subdir_path = os.path.join(video_path, subdir)
                frames = _collect_frames(subdir_path)
                if len(frames) > 0:
                    self.video_frames.append(frames)
                    self.video_names.append(subdir)

        # 构建扁平帧列表（用于 DataLoader 的长度计算）
        self.frames_paths = []
        for frames in self.video_frames:
            self.frames_paths.extend(frames)

        # 预计算每个帧所属的视频索引
        self.frame_to_video = []
        for vid_idx, frames in enumerate(self.video_frames):
            self.frame_to_video.extend([vid_idx] * len(frames))

        self.num_videos = len(self.video_frames)
        self.transform = transform
        self.same_identity_ratio = same_identity_ratio

        print(f'[Dataset] {self.num_videos} videos, {len(self.frames_paths)} total frames, '
              f'same_identity_ratio={same_identity_ratio}')
        for name, frames in zip(self.video_names, self.video_frames):
            print(f'  {name}: {len(frames)} frames')

    def __getitem__(self, idx):
        # 采样策略: same_identity_ratio 控制同身份 vs 跨身份比例
        # 同身份: source/target 来自同一视频 → 匹配推理分布 (推理时source和driving是同一人)
        # 跨身份: source/target 来自不同视频 → 保持泛化能力
        if self.num_videos >= 2 and random.random() < self.same_identity_ratio:
            # 同身份: 从同一视频中随机选择2帧
            vid_idx = random.choice(range(self.num_videos))
            frames = self.video_frames[vid_idx]
            if len(frames) >= 2:
                frame_a, frame_b = random.sample(frames, 2)
            else:
                frame_a = frame_b = random.choice(frames)
        elif self.num_videos >= 2:
            # 跨身份: 从不同视频各选1帧
            vid_a, vid_b = random.sample(range(self.num_videos), 2)
            frame_a = random.choice(self.video_frames[vid_a])
            frame_b = random.choice(self.video_frames[vid_b])
        else:
            # 单视频回退: 随机抽样
            nframes = len(self.frames_paths)
            items = random.sample(range(nframes), 2)
            frame_a = self.frames_paths[items[0]]
            frame_b = self.frames_paths[items[1]]

        img_source = Image.open(frame_a).convert('RGB')
        img_target = Image.open(frame_b).convert('RGB')

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.frames_paths)
