"""
SyncNet 训练脚本 (for EDTalk Audio2Mouth)

基于 HDTF 预处理数据训练 SyncNet，用于后续的 audio2mouth 同步损失监督。

数据结构要求（与 train_audio2mouth.py 一致）：
    {data_path}/
    ├── lmdb/              # 视频帧 LMDB 数据库
    ├── mel/               # 音频 mel 特征 (.npy)
    └── bbox/              # 人脸 bbox 坐标 (.npy)

训练输出：
    checkpoints/syncnet_*.pt  -- 训练好的 SyncNet 权重

用法示例：
    python train/train_syncnet.py --data_path HDTF --batch_size 64 --epochs 20
"""

import argparse
import os
import sys
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from io import BytesIO
import lmdb
import gc

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train.networks_audio2lip.syncnet import SyncNet_color as SyncNet
from train.networks_audio2lip.bilinear import crop_bbox_batch

# ---------------------------------------------------------------------------
# 数据加载相关（复用 dataset_audio2lip.py 中的工具函数）
# ---------------------------------------------------------------------------

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def _extract_video_names_from_env(env):
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


_lmdb_env_cache = {}


def _get_lmdb_env(lmdb_path):
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


# ---------------------------------------------------------------------------
# SyncNet Dataset
# ---------------------------------------------------------------------------

class SyncNetDataset(Dataset):
    """
    为 SyncNet 训练构建 (audio, face) 对。

    正样本对：音频帧与对应的视频帧（同步）
    负样本对：音频帧与随机偏移的视频帧（不同步）
    """

    def __init__(self, data_path, syncnet_T=5, image_size=256, val_split=0.05):
        """
        Args:
            data_path: 数据集根目录（包含 lmdb/, mel/, bbox/）
            syncnet_T: 时间窗口长度（帧数），默认 5，与 audio2mouth 训练一致
            image_size: 输入图片尺寸
            val_split: 验证集比例（默认 5%）
        """
        self.data_path = data_path
        self.syncnet_T = syncnet_T
        self.image_size = image_size
        self.half_T = syncnet_T // 2  # 中心帧偏移量

        # 打开 LMDB
        lmdb_path = os.path.join(data_path, 'lmdb')
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
        self.lmdb_path = lmdb_path
        self.env = None  # 延迟初始化，避免 pickle 问题

        # 提取视频列表（用临时 env）
        tmp_env = _get_lmdb_env(lmdb_path)
        self.video_list = _extract_video_names_from_env(tmp_env)

        # 预计算每个视频的有效帧数，构建索引
        all_samples = []  # (video_name, frame_idx) 列表
        self.mel_lengths = {}  # 缓存 mel 长度
        self.mel_cache = {}    # 缓存完整 mel 数据，避免 __getitem__ 重复磁盘读取
        self._orig_sizes = {}  # 缓存原始图像尺寸（用于 bbox 归一化）
        for video_name in self.video_list:
            mel_path = os.path.join(data_path, 'mel', video_name + '.npy')
            if not os.path.exists(mel_path):
                continue
            mel = np.load(mel_path)
            l = len(mel)
            self.mel_lengths[video_name] = l
            self.mel_cache[video_name] = mel
            for i in range(self.half_T, l - self.half_T):
                all_samples.append((video_name, i))

        # 划分训练集和验证集（按视频划分，避免数据泄漏）
        # 使用确定性随机种子，确保划分结果可复现
        rng = random.Random(42)
        shuffled_videos = self.video_list.copy()
        rng.shuffle(shuffled_videos)
        
        val_video_count = max(1, int(len(shuffled_videos) * val_split))
        self.val_videos = set(shuffled_videos[:val_video_count])
        self.train_videos = set(shuffled_videos[val_video_count:])
        
        self.train_samples = [s for s in all_samples if s[0] in self.train_videos]
        self.val_samples = [s for s in all_samples if s[0] in self.val_videos]
        
        # 默认使用训练集
        self.samples = self.train_samples
        
        print(f"[INFO] Total samples: {len(all_samples)} (train: {len(self.train_samples)}, val: {len(self.val_samples)})")
        print(f"[INFO] Train videos: {len(self.train_videos)}, Val videos: {len(self.val_videos)}")

    def set_mode(self, is_train=True):
        """切换数据集模式（训练/验证）"""
        self.samples = self.train_samples if is_train else self.val_samples

    def __getstate__(self):
        """排除不可 pickle 的 lmdb.Environment 对象"""
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_env(self):
        """延迟获取 LMDB 环境，每个 worker 进程独立持有"""
        if self.env is None:
            self.env = _get_lmdb_env(self.lmdb_path)
        return self.env

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, video_name, frame_idx, crop_lower_half=False):
        """从 LMDB 加载单帧图片并转换为 tensor (C, H, W)，范围 [-1, 1]
        
        Args:
            crop_lower_half: 是否只保留下半脸（默认 False，由外部统一裁剪）
        """
        with self._get_env().begin(write=False) as txn:
            key = format_for_lmdb(video_name, frame_idx)
            img_bytes = txn.get(key)
            if img_bytes is None:
                return None
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            img = img.resize((96, 96))
            img = np.array(img).astype(np.float32) / 255.0
            
            if crop_lower_half:
                h = img.shape[0]
                img = img[h//2:, :, :]
            
            img = (img - 0.5) / 0.5
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img

    def _load_mel(self, video_name, frame_idx, mel_data=None):
        """加载对应帧的 mel 特征切片

        Args:
            mel_data: 若已加载过该视频的完整 mel 数组，可直接传入以避免重复磁盘读取
        """
        if mel_data is None:
            mel_path = os.path.join(self.data_path, 'mel', video_name + '.npy')
            mel_data = np.load(mel_path)
        # 取 syncnet_T 帧对应的 mel
        start = frame_idx - self.half_T
        end = frame_idx + self.half_T + 1
        if start < 0 or end > len(mel_data):
            return None
        mel_slice = mel_data[start:end]  # (syncnet_T, 80, 16)
        return torch.from_numpy(mel_slice).float()

    def _load_bbox(self, video_name, frame_idx):
        """加载对应帧的 bbox"""
        bbox_path = os.path.join(self.data_path, 'bbox', video_name + '.npy')
        if not os.path.exists(bbox_path):
            return None
        bbox = np.load(bbox_path)
        if frame_idx >= len(bbox):
            return None
        return bbox[frame_idx]

    def _load_face_sequence(self, video_name, frame_idx):
        """加载以 frame_idx 为中心的 5 帧 face sequence，应用 bbox 裁剪 + 下半脸裁剪。
        与 audio2mouth 推理时 get_sync_loss 的处理流程完全一致。
        返回: (15, 48, 96) 或 None
        """
        half_T = self.half_T
        mel = self.mel_cache.get(video_name)
        if mel is None:
            return None
        mel_len = len(mel)
        
        # 边界检查
        if frame_idx < half_T or frame_idx >= mel_len - half_T:
            return None

        face_frames = []
        for t in range(frame_idx - half_T, frame_idx + half_T + 1):
            f = self._load_frame(video_name, t, crop_lower_half=False)
            if f is None:
                f = self._load_frame(video_name, frame_idx, crop_lower_half=False)
                if f is None:
                    return None
            face_frames.append(f)

        face_sequence = torch.cat(face_frames, dim=0)  # (15, 96, 96)

        # bbox 裁剪（对齐 audio2mouth 推理管线 get_sync_loss）
        # 推理时：crop_bbox_batch(recon_256x256, bbox/256, ..., 96) → 96×96 人脸
        # 训练时：需从 LMDB 原始图像中用相同逻辑裁剪
        # 关键修复：bbox 坐标在原始图像空间，需用原始图像尺寸归一化
        bbox = self._load_bbox(video_name, frame_idx)
        if bbox is not None:
            # 获取原始图像尺寸（首次访问时缓存，同视频所有帧尺寸相同）
            if video_name not in self._orig_sizes:
                with self._get_env().begin(write=False) as txn:
                    key = format_for_lmdb(video_name, frame_idx)
                    img_bytes = txn.get(key)
                    if img_bytes is not None:
                        orig_img = Image.open(BytesIO(img_bytes))
                        self._orig_sizes[video_name] = orig_img.size  # (W, H)
            orig_w, orig_h = self._orig_sizes.get(video_name, (96, 96))
            # 正确归一化：bbox / 原始图像尺寸 → [0, 1] 范围
            bbox_norm = torch.tensor([
                bbox[0] / orig_w, bbox[1] / orig_h,
                bbox[2] / orig_w, bbox[3] / orig_h
            ], dtype=torch.float32).unsqueeze(0)
            seq_batch = face_sequence.unsqueeze(0)
            box_to_feat = torch.tensor([0])
            try:
                cropped = crop_bbox_batch(seq_batch, bbox_norm, box_to_feat, 96)
                if cropped is not None:
                    face_sequence = cropped.squeeze(0)
            except Exception:
                pass

        # 下半脸裁剪: (15, 96, 96) → (15, 48, 96)
        h = face_sequence.shape[1]
        face_sequence = face_sequence[:, h // 2:, :]

        return face_sequence

    def __getitem__(self, idx):
        # 循环查找有效样本，避免递归
        max_attempts = 10
        for attempt in range(max_attempts):
            sample_idx = (idx + attempt) % len(self)
            video_name, frame_idx = self.samples[sample_idx]

            # 从缓存获取该视频的完整 mel 数据
            full_mel_data = self.mel_cache.get(video_name)
            if full_mel_data is None:
                continue

            # 加载 mel 特征 (syncnet_T, 80, 16)，复用已加载的 mel 数据
            mel = self._load_mel(video_name, frame_idx, mel_data=full_mel_data)
            if mel is None:
                continue

            # mel 合并为 (1, 80, 16 * syncnet_T) - SyncNet audio_encoder 期望 (B, 1, H, W)
            mel = mel.view(1, 80, -1)  # (1, 80, 80)

            # 加载正样本 face sequence
            face_sequence = self._load_face_sequence(video_name, frame_idx)
            if face_sequence is None:
                continue

            # 获取当前视频的有效帧范围（使用预缓存长度，不再重复加载 mel）
            mel_len = self.mel_lengths.get(video_name, frame_idx + 1)

            neg_face = None
            # 时间偏移负样本：同视频、至少偏移 3 帧的时刻。
            # 关键设计（对齐原始 Wav2Lip）：负样本必须是同一人物但不同时刻，
            # 这样模型学习的是"唇形是否与音频同步"（内容匹配），
            # 而非"人脸是否来自同一视频"（身份匹配）。
            # 偏移量 3~10 帧（120~400ms @ 25fps），覆盖爆破音
            # /b//p//m//f/ 的动态窗口（闭唇→释放 40~120ms），
            # 模型被迫学习精细口型差异。
            valid_hi = mel_len - self.half_T - 1  # 有效帧索引上界
            valid_lo = self.half_T                 # 有效帧索引下界
            shifts = []
            for d in range(3, 11):
                if valid_lo <= frame_idx - d <= valid_hi:
                    shifts.append(-d)
                if valid_lo <= frame_idx + d <= valid_hi:
                    shifts.append(d)
            if shifts:
                neg_frame_idx = frame_idx + random.choice(shifts)
                neg_face = self._load_face_sequence(video_name, neg_frame_idx)

            if neg_face is None:
                continue

            return {
                'mel': mel,                          # 锚定音频 mel (1, 80, 80)
                'face': face_sequence,               # 正样本人脸 (15, 48, 96)
                'neg_face': neg_face,                # 负样本人脸 (15, 48, 96) — 时间偏移（同视频不同时刻）
                'video_name': video_name,
                'frame_idx': frame_idx,
            }

        # 无法找到有效样本，返回空样本（不应该发生）
        raise RuntimeError(f"Cannot find valid sample after {max_attempts} attempts")


class _SyncNetDatasetView(SyncNetDataset):
    """SyncNetDataset 的轻量级视图，共享底层加载逻辑但持有独立的 samples 列表。

    用于解决 train_loader 和 val_loader 共享同一 dataset 对象时，
    set_mode() 切换 self.samples 导致 DataLoader.__len__() 动态计算错误的问题。

    PyTorch DataLoader.__len__() 的调用链:
        len(loader) -> len(BatchSampler) -> len(Sampler) -> len(dataset) -> len(dataset.samples)
    由于 __len__ 是动态计算的（非创建时缓存），若 train_loader 和 val_loader
    共享同一个 dataset 对象，set_mode() 会污染所有 loader 的长度计算结果。
    """

    def __init__(self, base: 'SyncNetDataset', samples):
        # 共享底层所有状态（lmdb_path, mel_lengths, video_list, half_T, syncnet_T 等）
        # 避免重复加载 mel 文件信息
        self.__dict__.update(base.__dict__)
        # 仅替换 samples 为指定的列表（train_samples 或 val_samples）
        self.samples = samples
        # env 不 pickle，每个 worker 进程独立初始化 LMDB
        self.env = None


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------

class SyncNetLoss(nn.Module):
    """
    联合损失：逐样本对比（主） + Batch 内 InfoNCE（辅助防坍缩）。

    设计原理：
    - 主损失 (loss_pos + loss_rel)：让音频和匹配人脸同步
    - 辅助 InfoNCE (0.05 权重)：防止 face_encoder 坍缩
      当 v_pos_emb 全部相同（坍缩）时 InfoNCE 会很大，
      迫使 face_encoder 为不同的人脸产生不同的 embedding。

    训练后的 SyncNet 可用于 audio2mouth 管线的 get_sync_loss：
    高 cos_sim 表示唇形与音频同步，低 cos_sim 表示不同步。
    """

    def __init__(self, margin=0.3, tau=0.1, aux_weight=0.05):
        super(SyncNetLoss, self).__init__()
        self.margin = margin
        self.tau = tau
        self.aux_weight = aux_weight

    def forward(self, a_emb, v_pos_emb, v_neg_emb):
        """
        Args:
            a_emb:      (B, D) L2-normalized audio embeddings
            v_pos_emb:  (B, D) L2-normalized positive face embeddings
            v_neg_emb:  (B, D) L2-normalized negative face embeddings
        Returns:
            loss / pos_sim / neg_sim
        """
        B = a_emb.size(0)
        pos_sim = (a_emb * v_pos_emb).sum(dim=1)  # (B,)
        neg_sim = (a_emb * v_neg_emb).sum(dim=1)  # (B,)

        # 主损失：逐样本对比
        loss_pos = (1.0 - pos_sim).pow(2).mean()
        loss_rel = F.relu(neg_sim - pos_sim + self.margin).pow(2).mean()
        loss_main = loss_pos + loss_rel

        # 辅助 InfoNCE：防止 face_encoder 坍缩
        sim_matrix = torch.mm(a_emb, v_pos_emb.t()) / self.tau
        labels = torch.arange(B, dtype=torch.long, device=sim_matrix.device)
        loss_aux = F.cross_entropy(sim_matrix, labels)

        loss = loss_main + self.aux_weight * loss_aux

        return loss, pos_sim.mean(), neg_sim.mean()


# ---------------------------------------------------------------------------
# 训练工具函数
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path):
    """保存 SyncNet checkpoint"""
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    # type: (str, nn.Module, torch.optim.Optimizer|None, torch.optim.lr_scheduler|None, str) -> int
    """加载 SyncNet checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        return 0

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    epoch = ckpt.get('epoch', 0)
    print(f"[INFO] Loaded checkpoint from epoch {epoch}")
    return epoch


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个 epoch，返回 (平均损失, 平均正样本相似度, 平均负样本相似度)"""
    model.train()
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        mel = batch['mel'].to(device)              # (B, 1, 80, 16*T)
        face = batch['face'].to(device)            # (B, 15, H, W)  — 同步人脸（正样本）
        neg_face = batch['neg_face'].to(device)    # (B, 15, H, W)  — 时间偏移人脸（负样本，同视频不同时刻）

        batch_size = mel.size(0)

        if batch_size < 2:
            continue

        # 音频编码器
        audio_feat = model.audio_encoder(mel)
        audio_feat = model.audio_pool(audio_feat)
        a_emb = audio_feat.view(batch_size, -1)
        a_emb = F.normalize(a_emb, p=2, dim=1)

        # 人脸编码器（正样本：同视频同时刻）
        face_feat = model.face_encoder(face)
        face_feat = model.face_pool(face_feat)
        v_pos_emb = face_feat.view(batch_size, -1)
        v_pos_emb = F.normalize(v_pos_emb, p=2, dim=1)

        # 人脸编码器（负样本：时间偏移，同视频不同时刻）
        neg_feat = model.face_encoder(neg_face)
        neg_feat = model.face_pool(neg_feat)
        v_neg_emb = neg_feat.view(batch_size, -1)
        v_neg_emb = F.normalize(v_neg_emb, p=2, dim=1)

        # ---- 诊断：验证数据流正确性 ----
        if batch_idx % 100 == 0:
            # 检查1: face vs neg_face 的原始像素差异
            face_diff = (face - neg_face).abs().mean().item()
            # 检查2: embedding 差异
            emb_diff = (v_pos_emb - v_neg_emb).abs().mean().item()
            # 检查3: 同一样本的 pos vs neg embedding 的 cosine 距离
            per_sample_cos_diff = (v_pos_emb * v_neg_emb).sum(dim=1).mean().item()
            # 检查4: pos_emb 自身的标准差（是否所有 pos_emb 都一样）
            pos_emb_std = v_pos_emb.std(dim=0).mean().item()
            # 检查5: audio_emb 的 batch 内标准差
            audio_emb_std = a_emb.std(dim=0).mean().item()
            # 检查6: Python 对象是否是同一个 tensor（内存地址相同）
            same_obj = (face.data_ptr() == neg_face.data_ptr())
            print(f"[DIAG] batch {batch_idx}: "
                  f"same_ptr={same_obj} | "
                  f"face_raw_diff={face_diff:.6f} | "
                  f"emb_diff={emb_diff:.6f} | "
                  f"pos_neg_cos={per_sample_cos_diff:.4f} | "
                  f"pos_std={pos_emb_std:.4f} | "
                  f"audio_std={audio_emb_std:.4f}")

        # Wav2Lip 式逐样本对比损失（不使用 batch 内 InfoNCE）
        loss, pos_sim, neg_sim = criterion(a_emb, v_pos_emb, v_neg_emb)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_pos_sim += pos_sim.item() * batch_size
        total_neg_sim += neg_sim.item() * batch_size
        total_samples += batch_size

        if batch_idx % 100 == 0:
            avg_loss = total_loss / total_samples
            avg_pos_sim = total_pos_sim / total_samples
            avg_neg_sim = total_neg_sim / total_samples
            print(f"[Train] Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.6f} | Pos Sim: {avg_pos_sim:.4f} | Neg Sim: {avg_neg_sim:.4f}")

    avg_loss = total_loss / total_samples
    avg_pos_sim = total_pos_sim / total_samples
    avg_neg_sim = total_neg_sim / total_samples

    return avg_loss, avg_pos_sim, avg_neg_sim


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device, epoch):
    """验证一个 epoch，返回 (平均损失, 平均正样本相似度, 平均负样本相似度)"""
    model.eval()
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        mel = batch['mel'].to(device)
        face = batch['face'].to(device)
        neg_face = batch['neg_face'].to(device)    # 时间偏移人脸（负样本）

        batch_size = mel.size(0)

        if batch_size < 2:
            continue

        audio_feat = model.audio_encoder(mel)
        audio_feat = model.audio_pool(audio_feat)
        a_emb = audio_feat.view(batch_size, -1)
        a_emb = F.normalize(a_emb, p=2, dim=1)

        face_feat = model.face_encoder(face)
        face_feat = model.face_pool(face_feat)
        v_pos_emb = face_feat.view(batch_size, -1)
        v_pos_emb = F.normalize(v_pos_emb, p=2, dim=1)

        neg_feat = model.face_encoder(neg_face)
        neg_feat = model.face_pool(neg_feat)
        v_neg_emb = neg_feat.view(batch_size, -1)
        v_neg_emb = F.normalize(v_neg_emb, p=2, dim=1)

        loss, pos_sim, neg_sim = criterion(a_emb, v_pos_emb, v_neg_emb)

        total_loss += loss.item() * batch_size
        total_pos_sim += pos_sim.item() * batch_size
        total_neg_sim += neg_sim.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_pos_sim = total_pos_sim / total_samples
    avg_neg_sim = total_neg_sim / total_samples

    print(f"[Eval]  Epoch {epoch} Loss: {avg_loss:.6f} | Pos Sim: {avg_pos_sim:.4f} | Neg Sim: {avg_neg_sim:.4f}")
    return avg_loss, avg_pos_sim, avg_neg_sim


def main(args):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(args.checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 数据集（划分训练/验证集）
    print("[INFO] Loading datasets...")

    # 处理 --no_val_split：使用全量数据集训练（不划分验证集）
    if args.no_val_split:
        # 临时创建 dataset 获取所有样本
        temp_dataset = SyncNetDataset(
            args.data_path,
            syncnet_T=args.syncnet_T,
            image_size=args.image_size,
            val_split=0.0  # 不划分验证集
        )
        # 使用全部样本作为训练集
        all_samples = temp_dataset.train_samples + temp_dataset.val_samples
        base_dataset = temp_dataset
        train_dataset = _SyncNetDatasetView(base_dataset, all_samples)
        val_dataset = None
        print(f"[INFO] Using FULL dataset (no val split): {len(all_samples)} samples")
    else:
        # 标准模式：按 val_split 划分训练/验证集
        base_dataset = SyncNetDataset(
            args.data_path,
            syncnet_T=args.syncnet_T,
            image_size=args.image_size,
            val_split=args.val_split
        )
        # 创建独立的训练/验证视图，避免共享同一 dataset 对象导致 set_mode 状态污染。
        # PyTorch DataLoader.__len__() 是动态计算的（非创建时缓存），若共享同一个
        # dataset 对象，后调用的 set_mode 会污染所有 loader 的长度和采样索引。
        # _SyncNetDatasetView 共享底层 LMDB/mel 缓存，但持有独立的 samples 列表。
        train_dataset = _SyncNetDatasetView(base_dataset, base_dataset.train_samples)
        val_dataset = _SyncNetDatasetView(base_dataset, base_dataset.val_samples)

    # 训练集
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False
    )

    # 验证集（仅在非 no_val_split 模式时创建）
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            drop_last=False
        )
    else:
        val_loader = None

    print(f"[INFO] Train batches: {len(train_loader)}" +
          (f", Val batches: {len(val_loader)}" if val_loader else ", Val: DISABLED"))

    # 模型
    print("[INFO] Initializing SyncNet...")
    model = SyncNet().to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = SyncNetLoss()

    # 学习率调度：ReduceLROnPlateau，监控 Val Gap 的改善
    # 传递 -Gap 给 mode='min'（因为 Gap 增大 = 需最小化 -Gap）
    # threshold_mode='abs' + threshold=0.003: 仅当 Gap 改善 ≥0.003 时才重置计数
    # 这过滤了 ±0.001~0.002 的随机波动，确保 LR 只在真实停滞时衰减
    lr_min_effective = max(args.lr_min, args.lr / 20)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=lr_min_effective,
        threshold_mode='abs',
        threshold=0.003,  # Gap 改善 < 0.003 → 视为未改善
    )

    # 加载已有 checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, str(device))
        start_epoch += 1

    # Early Stopping 参数
    es_patience = args.es_patience
    es_counter = 0           # Gap 未改善的连续 epoch 数
    best_save_epoch = -1     # 最后一次保存 best model 的 epoch

    # 训练循环
    # 最佳模型追踪：使用 Val Pos/Neg Gap 而非 val_loss
    # val_loss 波动大且与下游 audio2lip 任务关联弱，Gap 直接反映
    # 模型的唇音同步判别能力——这才是 SyncNet 的核心指标。
    best_val_loss = float('inf')
    best_val_gap = -float('inf')   # 追踪最佳 Pos/Neg 分离度
    initial_lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.8f}")
        print(f"{'='*60}")

        # 训练并获取训练集指标
        train_loss, train_pos_sim, train_neg_sim = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # 验证并获取验证集指标（仅在非 no_val_split 模式时）
        if val_loader is not None:
            val_loss, val_pos_sim, val_neg_sim = eval_epoch(
                model, val_loader, criterion, device, epoch
            )
            val_gap = val_pos_sim - val_neg_sim

            # 打印 epoch 结束后的完整指标
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Val Pos Sim: {val_pos_sim:.4f} | Val Neg Sim: {val_neg_sim:.4f} | "
                  f"Val Gap: {val_gap:.4f}")

            # TensorBoard 记录
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Sim/pos_train', train_pos_sim, epoch)
            writer.add_scalar('Sim/neg_train', train_neg_sim, epoch)
            writer.add_scalar('Sim/pos_val', val_pos_sim, epoch)
            writer.add_scalar('Sim/neg_val', val_neg_sim, epoch)
            writer.add_scalar('Gap/val', val_gap, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # 保存 checkpoint（每5个epoch保存一次）
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'syncnet_epoch_{epoch:04d}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)

            # ---- 保存最佳模型：双标准 ----
            # 标准1（主）：Val Pos/Neg Gap 最大 —— 反映唇音判别力
            # 标准2（辅）：若 Gap 相同，Val Pos Sim 更高优先
            is_new_best = False
            best_reason = ""
            if val_gap >= best_val_gap + 0.003:  # 微小容差，过滤波动但保留所有有意义改善
                is_new_best = True
                best_reason = f"Val Gap: {val_gap:.4f} (prev {best_val_gap:.4f})"
            elif abs(val_gap - best_val_gap) <= 0.005 and val_pos_sim > best_val_loss + 0.01:
                # Gap 接近时，Pos Sim 更高的模型在推理时更可靠
                is_new_best = True
                best_reason = f"Val Pos: {val_pos_sim:.4f} (prev best at same Gap)"
                best_val_loss = val_pos_sim  # 复用 best_val_loss 存储 best_pos

            if is_new_best:
                best_val_gap = val_gap
                if 'Pos' not in best_reason:
                    best_val_loss = val_pos_sim
                best_path = os.path.join(args.checkpoint_dir, 'syncnet_best.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
                print(f"[INFO] New best model saved! {best_reason}")
                es_counter = 0
                best_save_epoch = epoch
            else:
                es_counter += 1

            # 学习率调度：监控 Val Gap（Gap 增大=改善，传递 -Gap 给 mode='min'）
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(-val_gap)
            current_lr = optimizer.param_groups[0]['lr']

            # LR 衰减时输出提示
            if current_lr < prev_lr:
                print(f"[LR] Learning rate decayed: {prev_lr:.8f} → {current_lr:.8f}")

            # ---- Early Stopping（双条件） ----
            # 条件1：Gap 连续 N epoch 未改善 → 触发早停
            if es_patience > 0 and es_counter >= es_patience:
                print(f"[Early Stop] Val Gap not improved for {es_patience} epochs "
                      f"(since epoch {best_save_epoch}). Best Val Gap: {best_val_gap:.4f}")
                break

            # 条件2：LR 已衰减到初始 LR 的 1/10 以下 → 训练无效，停止
            if current_lr < initial_lr / 10:
                print(f"[Early Stop] LR ({current_lr:.8f}) below useful threshold "
                      f"({initial_lr/10:.8f}). Best Val Gap: {best_val_gap:.4f}")
                break
        else:
            # no_val_split 模式：基于训练损失保存 checkpoint
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | "
                  f"Train Pos Sim: {train_pos_sim:.4f} | Train Neg Sim: {train_neg_sim:.4f}")

            # TensorBoard 记录（仅训练指标）
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Sim/pos_train', train_pos_sim, epoch)
            writer.add_scalar('Sim/neg_train', train_neg_sim, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # 保存 checkpoint（每5个epoch保存一次）
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'syncnet_epoch_{epoch:04d}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_path)

            # 基于训练损失早停（连续多个 epoch 训练损失未下降）
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_path = os.path.join(args.checkpoint_dir, 'syncnet_best.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss, best_path)
                print(f"[INFO] New best model saved! Train Loss: {train_loss:.6f}")
                es_counter = 0
            else:
                es_counter += 1

            # 学习率衰减：在 no_val_split 模式下不使用（基于训练损失无意义）

            # Early Stopping
            if es_patience > 0 and es_counter >= es_patience:
                print(f"[Early Stop] No improvement for {es_patience} epochs. "
                      f"Best Train Loss: {best_val_loss:.6f}")
                break

        # 定期清理缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    writer.close()
    print("[INFO] Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SyncNet for EDTalk Audio2Mouth')

    # 数据路径
    parser.add_argument('--data_path', type=str, default='HDTF',
                        help='数据集根目录（包含 lmdb/, mel/, bbox/）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau 衰减因子')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='ReduceLROnPlateau patience（epoch 数）')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='最小学习率')
    parser.add_argument('--es_patience', type=int, default=25,
                        help='Early Stopping patience（epoch 数，0 表示禁用）')
    parser.add_argument('--syncnet_T', type=int, default=5,
                        help='SyncNet 时间窗口长度（帧数）')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图片尺寸')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker 数量')

    # 输出与恢复
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt_models/syncnet',
                        help='checkpoint 保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的 checkpoint 路径')
    parser.add_argument('--val_split', type=float, default=0.05,
                        help='验证集比例（按视频划分），默认 5%%')
    parser.add_argument('--no_val_split', action='store_true',
                        help='禁用验证集划分，使用全量数据集进行训练（适用于特定人物微调场景）')

    args = parser.parse_args()
    main(args)
