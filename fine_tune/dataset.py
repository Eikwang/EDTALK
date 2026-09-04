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


def _mouth_weight_map(fm68, mouth_weight=4.0, dilate_px=3, size=256):
    """target 帧的嘴区权重图 (1,size,size) float32。

    嘴区（外唇 48-59 凸包外扩 dilate_px）= mouth_weight，其余 = 1.0
    （背景保持 ×1.0 — 计划 H2：背景锚定是压制 direction 膨胀的约束，
    降权会反噬背景 SSIM 诊断门槛）。fm68 含 NaN（检测失败行）→ 全 1 回退。
    """
    import cv2
    import numpy as np
    w = np.ones((size, size), dtype=np.float32)
    if fm68 is not None and not np.isnan(fm68).any():
        m = np.zeros((size, size), dtype=np.uint8)
        pts = fm68[48:60].astype(np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
        if dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            m = cv2.dilate(m, k)
        w[m == 255] = mouth_weight
    return w[None, :, :]  # (1,H,W)


class Finetune256(Dataset):
    """微调数据集（FINE_TUNE_FIX_PLAN T4/B：采样对齐推理分布）。

    四路互斥采样（替换旧版 same_identity_ratio 二元策略）：
    - 跨视频远对 (far_ratio=0.3, ≥2 视频时): 同角色不同场次 —
      口型/姿势差覆盖面最大，保持 lip alpha 方差
      （近邻会收缩训练分布，与 Audio2Lip 全量程脱节）
    - 同帧自对 (self_ratio=0.1): source==target — 显式训练推理时的
      "pose alpha 来自自身"恒等条件（根因 R1 修复的最高杠杆项）
    - 时序近邻 (near_ratio=0.6): target = source 邻域 ±near_window
      （同视频、clamp 在采样池内）— 姿势差小、口型差大，即实时推理分布
      （render_loop.py:727 pose drive=source 自身，lip 由音频驱动）
    - 同视频远对 (兜底, 比例和 <1 时): 全池随机两帧

    held-out: 每视频末 held_out_ratio 比例帧从**所有**采样路径剔除
    （与 tools/eval_finetune.py 的划分口径一致 — 邻域窗口若采到评估帧
    当 target，指标静默虚高、决策门失效）。

    权重图: 返回 target 帧嘴区权重 (1,256,256)，供 vgg/l1 嘴区加权（方案 H）。
    facemark npy 行数与帧数不符 → fail-fast（缓存与帧序错位禁止静默训练）。

    单帧/空池回退: 池内可用帧 <2 时退化为自对（旧版在此崩溃）。
    """

    def __init__(self, video_path, transform=None, facemark_dir=None,
                 near_window=64, near_ratio=0.6, self_ratio=0.1,
                 far_ratio=0.3, held_out_ratio=0.1, mouth_weight=4.0,
                 same_identity_ratio=None, **kwargs):
        if same_identity_ratio is not None:
            print('[Dataset] same_identity_ratio 已废弃 — 使用 '
                  'near/self/far 四路采样 (计划 T4/B)')
        img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

        def _collect_frames(path):
            files = []
            for ext in img_exts:
                files.extend(glob.glob(os.path.join(path, ext)))
            return _sort_frame_files(files)

        self.video_frames = []     # [[video1_frames], ...]
        self.video_names = []
        self.video_facemarks = []  # np.ndarray (N,68,2) or None per video

        direct_frames = _collect_frames(video_path)
        if direct_frames:
            self.video_frames.append(direct_frames)
            self.video_names.append(os.path.basename(os.path.abspath(video_path)))
        else:
            subdirs = sorted(f for f in os.listdir(video_path)
                             if os.path.isdir(os.path.join(video_path, f)))
            for d in subdirs:
                frames = _collect_frames(os.path.join(video_path, d))
                if frames:
                    self.video_frames.append(frames)
                    self.video_names.append(d)

        # facemark 加载（嘴区权重图来源；默认 datapath 同级 facemark/ 目录）
        import numpy as np
        if facemark_dir is None:
            parent = os.path.dirname(os.path.abspath(video_path))
            for cand in (os.path.join(parent, 'facemark'),
                         os.path.join(video_path, 'facemark')):
                if os.path.isdir(cand):
                    facemark_dir = cand
                    break
        for name, frames in zip(self.video_names, self.video_frames):
            fm = None
            if facemark_dir:
                npy = os.path.join(facemark_dir, f'{name}.npy')
                if os.path.exists(npy):
                    fm = np.load(npy)
                    if len(fm) != len(frames):
                        raise RuntimeError(
                            f'[{name}] facemark 行数 {len(fm)} != 帧数 {len(frames)}'
                            f' — 缓存与帧序错位, 禁止静默训练 (计划 A2)')
            self.video_facemarks.append(fm)

        # held-out 剔除: 每视频采样池 = [0, n - hold_n)（与 eval 口径一致）
        self.pools = []          # 每视频可用帧索引列表（视频内帧序号）
        self.held_out_info = []  # (lo, n) or None
        for frames in self.video_frames:
            n = len(frames)
            hold_n = max(1, int(n * held_out_ratio)) if n >= 2 else 0
            pool = list(range(n - hold_n)) if hold_n else list(range(n))
            self.pools.append(pool)
            self.held_out_info.append((n - hold_n, n) if hold_n else None)
        self.frames_paths = [f for frames in self.video_frames for f in frames]

        self.near_window = max(1, int(near_window))
        self.near_ratio = float(near_ratio)
        self.self_ratio = float(self_ratio)
        self.far_ratio = float(far_ratio)
        self.transform = transform
        self.mouth_weight = float(mouth_weight)

        total_pool = sum(len(p) for p in self.pools)
        n_held = sum((e[1] - e[0]) for e in self.held_out_info if e is not None)
        print(f'[Dataset] {len(self.video_frames)} videos, '
              f'{len(self.frames_paths)} frames, 采样池 {total_pool}, '
              f'held-out 剔除 {n_held} (末 {held_out_ratio:.0%})')
        print(f'[Dataset] 采样: near(±{self.near_window})={self.near_ratio:.0%} '
              f'self={self.self_ratio:.0%} far={self.far_ratio:.0%}; '
              f'mouth_weight={self.mouth_weight}')

    def _pick_pool_frame(self, vid):
        pool = self.pools[vid]
        if not pool:
            raise RuntimeError(f'[{self.video_names[vid]}] 采样池为空')
        return pool[random.randrange(len(pool))]

    def __getitem__(self, idx):
        import numpy as np
        nv = len(self.video_frames)
        if nv == 0:
            raise RuntimeError('数据集为空')
        r = random.random()
        frame_a = frame_b = None
        fm_b = None

        if r < self.far_ratio and nv >= 2:
            # 跨视频远对: 同角色不同场次
            va, vb = random.sample(range(nv), 2)
            ia = self._pick_pool_frame(va)
            ib = self._pick_pool_frame(vb)
            frame_a = self.video_frames[va][ia]
            frame_b = self.video_frames[vb][ib]
            fmb_arr = self.video_facemarks[vb]   # target 在 vb — facemark 也取 vb
            fm_b = None if fmb_arr is None else fmb_arr[ib]
        else:
            vid = random.randrange(nv)
            pool = self.pools[vid]
            frames = self.video_frames[vid]
            fm_arr = self.video_facemarks[vid]
            if len(pool) < 2:
                # 单帧回退（旧版 random.sample(range(1),2) 在此崩溃 — 已修）
                i = pool[0]
                frame_a = frame_b = frames[i]
                fm_b = None if fm_arr is None else fm_arr[i]
            else:
                i = self._pick_pool_frame(vid)
                r2 = random.random()
                if r2 < self.self_ratio:
                    # 自对: source==target（恒等条件）
                    frame_a = frame_b = frames[i]
                    fm_b = None if fm_arr is None else fm_arr[i]
                elif r2 < self.self_ratio + self.near_ratio:
                    # 近邻: ±near_window, clamp 在池内（不出 held-out）
                    d = random.randint(1, self.near_window)
                    if random.random() < 0.5:
                        d = -d
                    j = min(max(i + d, pool[0]), pool[-1])
                    if j == i:
                        j = min(max(i + 1, pool[0]), pool[-1])
                    frame_a, frame_b = frames[i], frames[j]
                    fm_b = None if fm_arr is None else fm_arr[j]
                else:
                    # 同视频远对: 全池随机（near+self+far<1 时的兜底路）
                    j = self._pick_pool_frame(vid)
                    while j == i and len(pool) > 1:
                        j = self._pick_pool_frame(vid)
                    frame_a, frame_b = frames[i], frames[j]
                    fm_b = None if fm_arr is None else fm_arr[j]

        img_source = Image.open(frame_a).convert('RGB')
        img_target = Image.open(frame_b).convert('RGB')
        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        # 权重图基于 target 帧嘴区（监督目标是 target）
        import torch
        weight = torch.from_numpy(
            _mouth_weight_map(fm_b, mouth_weight=self.mouth_weight))
        return img_source, img_target, weight

    def __len__(self):
        return len(self.frames_paths)
