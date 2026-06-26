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
        self.mel_lengths = {}  # 缓存 mel 长度，避免 __getitem__ 中重复磁盘读取
        for video_name in self.video_list:
            mel_path = os.path.join(data_path, 'mel', video_name + '.npy')
            if not os.path.exists(mel_path):
                continue
            mel = np.load(mel_path)
            # 有效帧范围：需要前后各 half_T 帧，且 mel 特征足够
            l = len(mel)
            self.mel_lengths[video_name] = l
            # 跳过边缘帧
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

    def _load_frame(self, video_name, frame_idx, crop_lower_half=True):
        """从 LMDB 加载单帧图片并转换为 tensor (C, H, W)，范围 [-1, 1]
        
        Args:
            crop_lower_half: 是否只保留下半脸（与 audio2mouth 训练时的 get_sync_loss 一致）
        """
        with self._get_env().begin(write=False) as txn:
            key = format_for_lmdb(video_name, frame_idx)
            img_bytes = txn.get(key)
            if img_bytes is None:
                return None
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            # SyncNet face_encoder 期望输入 (B, 15, 48, 96)
            # 但在 audio2mouth 的 get_sync_loss 中，人脸被裁剪为下半部分 (48, 96)
            # 因此这里加载 96x96 后裁剪下半部分，确保训练/推理一致性
            img = img.resize((96, 96))
            img = np.array(img).astype(np.float32) / 255.0
            
            if crop_lower_half:
                # 只保留下半脸: H//2: (48, 96, 3)
                # 注意: numpy 数组是 (H, W, C)，所以 axis 0 是 H
                h = img.shape[0]
                img = img[h//2:, :, :]
            
            # 归一化到 [-1, 1]
            img = (img - 0.5) / 0.5
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
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

    def __getitem__(self, idx):
        # 循环查找有效样本，避免递归
        max_attempts = 10
        for attempt in range(max_attempts):
            sample_idx = (idx + attempt) % len(self)
            video_name, frame_idx = self.samples[sample_idx]

            # 预加载该视频的完整 mel 数据（避免同一样本重复磁盘读取）
            mel_path = os.path.join(self.data_path, 'mel', video_name + '.npy')
            try:
                full_mel_data = np.load(mel_path)
            except Exception:
                continue

            # 加载 mel 特征 (syncnet_T, 80, 16)，复用已加载的 mel 数据
            mel = self._load_mel(video_name, frame_idx, mel_data=full_mel_data)
            if mel is None:
                continue

            # mel 合并为 (1, 80, 16 * syncnet_T) - SyncNet audio_encoder 期望 (B, 1, H, W)
            mel = mel.view(1, 80, -1)  # (1, 80, 80)

            # 加载同步的视频帧（正样本）
            img = self._load_frame(video_name, frame_idx)
            if img is None:
                continue

            # 加载 bbox 并裁剪下半脸（与 audio2mouth 训练一致）
            bbox = self._load_bbox(video_name, frame_idx)
            if bbox is None:
                # 如果无 bbox，使用整张图片
                face_img = img
            else:
                # bbox 格式: [x1, y1, x2, y2]
                # crop_bbox_batch 需要 batch 输入
                bbox_tensor = torch.from_numpy(bbox).float().unsqueeze(0)  # (1, 4)
                img_batch = img.unsqueeze(0)  # (1, 3, H, W)
                box_to_feat = torch.tensor([0])
                try:
                    face_img = crop_bbox_batch(img_batch, bbox_tensor / 96, box_to_feat, 96)
                    if face_img is not None:
                        face_img = face_img.squeeze(0)  # (3, 96, 96)
                    else:
                        face_img = img
                except Exception:
                    face_img = img

            # 构建正样本对（同步）
            # face_encoder 期望输入: (B, 15, H, W) 即 5帧 x 3通道
            # 这里我们取中心帧前后各 half_T 帧
            face_frames = []
            for t in range(frame_idx - self.half_T, frame_idx + self.half_T + 1):
                f = self._load_frame(video_name, t)
                if f is None:
                    f = img  # 回退到中心帧
                face_frames.append(f)

            # 拼接为 (15, H, W) -> 5帧 * 3通道
            face_sequence = torch.cat(face_frames, dim=0)  # (15, H, W)

            # 获取当前视频的有效帧范围（使用预缓存长度，不再重复加载 mel）
            mel_len = self.mel_lengths.get(video_name, frame_idx + 1)

            # 负样本策略：偏移范围 [2, 6] 帧（约 0.08-0.24秒）
            # 太小会让任务太简单（~0.08s），太大会让任务太难（>0.3s）
            neg_offset = random.randint(2, 6)  # [2, 6]
            if random.random() < 0.5:
                neg_offset = -neg_offset
            neg_frame_idx = frame_idx + neg_offset

            # 确保在有效范围内（使用视频实际帧数）
            neg_frame_idx = max(self.half_T, min(neg_frame_idx, mel_len - self.half_T - 1))

            # 复用已加载的 mel 数据，避免重复磁盘读取
            neg_mel = self._load_mel(video_name, neg_frame_idx, mel_data=full_mel_data)
            if neg_mel is None:
                neg_mel = mel  # 回退
            else:
                neg_mel = neg_mel.view(1, 80, -1)

            return {
                'mel': mel,                          # 正样本 mel (1, 80, 80)
                'face': face_sequence,               # 正样本 face (15, 96, 96)
                'neg_mel': neg_mel,                  # 负样本 mel (1, 80, 80)
                'video_name': video_name,
                'frame_idx': frame_idx,
            }

        # 无法找到有效样本，返回空样本（不应该发生）
        raise RuntimeError(f"Cannot find valid sample after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------

class SyncNetLoss(nn.Module):
    """
    SyncNet 对比损失。

    正样本对 (同步): 期望 cosine_similarity -> 1
    负样本对 (不同步): 期望 cosine_similarity -> 0

    使用 BCEWithLogitsLoss 稳定训练。
    """

    def __init__(self):
        super(SyncNetLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, a_emb, v_emb, neg_a_emb=None, neg_v_emb=None):
        """
        Args:
            a_emb: 正样本音频嵌入 (B, D)
            v_emb: 正样本视频嵌入 (B, D)
            neg_a_emb: 负样本音频嵌入 (B, D)，可选
            neg_v_emb: 负样本视频嵌入 (B, D)，可选
        Returns:
            loss: 标量损失
        """
        # 正样本对的 cosine similarity
        pos_sim = F.cosine_similarity(a_emb, v_emb, dim=1)  # (B,)
        # 映射到 logits: (sim + 1) / 2 -> [0,1], 然后 logit
        pos_logits = pos_sim * 5.0  # 缩放因子，使梯度更明显

        # 正样本标签为 1
        pos_labels = torch.ones_like(pos_logits)
        loss = self.bce(pos_logits, pos_labels)

        # 负样本对（如果提供）
        if neg_a_emb is not None and neg_v_emb is not None:
            neg_sim = F.cosine_similarity(neg_a_emb, neg_v_emb, dim=1)
            neg_logits = neg_sim * 5.0
            neg_labels = torch.zeros_like(neg_logits)
            loss = loss + self.bce(neg_logits, neg_labels)

        return loss


# ---------------------------------------------------------------------------
# 训练工具函数
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """保存 SyncNet checkpoint"""
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    # type: (str, nn.Module, torch.optim.Optimizer|None, str) -> int
    """加载 SyncNet checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        return 0

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
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
        face = batch['face'].to(device)            # (B, 15, H, W)
        neg_mel = batch['neg_mel'].to(device)      # (B, 1, 80, 16*T)

        batch_size = mel.size(0)

        # 前向传播
        a_emb, v_emb = model(mel, face)

        # 负样本对（使用负样本 mel + 正样本 face）
        neg_a_emb, _ = model(neg_mel, face)

        # 计算损失
        loss = criterion(a_emb, v_emb, neg_a_emb, v_emb)

        # 计算相似度用于监控
        pos_sim = F.cosine_similarity(a_emb, v_emb, dim=1).sum().item()
        neg_sim = F.cosine_similarity(neg_a_emb, v_emb, dim=1).sum().item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_pos_sim += pos_sim
        total_neg_sim += neg_sim
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
        neg_mel = batch['neg_mel'].to(device)

        batch_size = mel.size(0)

        a_emb, v_emb = model(mel, face)
        neg_a_emb, _ = model(neg_mel, face)
        loss = criterion(a_emb, v_emb, neg_a_emb, v_emb)

        pos_sim = F.cosine_similarity(a_emb, v_emb, dim=1).sum().item()
        neg_sim = F.cosine_similarity(neg_a_emb, v_emb, dim=1).sum().item()

        total_loss += loss.item() * batch_size
        total_pos_sim += pos_sim
        total_neg_sim += neg_sim
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
    dataset = SyncNetDataset(
        args.data_path,
        syncnet_T=args.syncnet_T,
        image_size=args.image_size,
        val_split=args.val_split
    )

    # 训练集
    dataset.set_mode(is_train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False
    )

    # 验证集
    dataset.set_mode(is_train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False
    )

    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 模型
    print("[INFO] Initializing SyncNet...")
    model = SyncNet().to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 损失函数
    criterion = SyncNetLoss()

    # 加载已有 checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch += 1

    # 学习率调度：连续 lr_patience 个 epoch 训练损失未创新低则降低 10%
    lr_patience = 5
    lr_decay_factor = 0.9
    min_lr = 1e-6  # 最小学习率保护
    epochs_since_improvement = 0

    # Early Stopping
    es_patience = 10
    es_counter = 0

    # 训练循环
    best_val_loss = float('inf')
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

        # 验证并获取验证集指标
        val_loss, val_pos_sim, val_neg_sim = eval_epoch(
            model, val_loader, criterion, device, epoch
        )

        # 打印 epoch 结束后的完整指标
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Val Pos Sim: {val_pos_sim:.4f} | Val Neg Sim: {val_neg_sim:.4f}")

        # TensorBoard 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Sim/pos_train', train_pos_sim, epoch)
        writer.add_scalar('Sim/neg_train', train_neg_sim, epoch)
        writer.add_scalar('Sim/pos_val', val_pos_sim, epoch)
        writer.add_scalar('Sim/neg_val', val_neg_sim, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 保存 checkpoint（每5个epoch保存一次）
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'syncnet_epoch_{epoch:04d}.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # 保存最佳模型 & 学习率调度 & Early Stopping（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'syncnet_best.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"[INFO] New best model saved! Val Loss: {val_loss:.6f}")
            epochs_since_improvement = 0
            es_counter = 0
        else:
            epochs_since_improvement += 1
            es_counter += 1

            # 学习率衰减：连续 lr_patience 个 epoch 未改善则降低 10%
            if epochs_since_improvement >= lr_patience:
                old_lr = optimizer.param_groups[0]['lr']
                if old_lr > min_lr:
                    new_lr = max(old_lr * lr_decay_factor, min_lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"[LR] No improvement for {lr_patience} epochs, "
                          f"lr: {old_lr:.8f} -> {new_lr:.8f}")
                    epochs_since_improvement = 0  # 重置计数，等待下一个 lr_patience 窗口
                else:
                    print(f"[LR] Already at minimum lr ({min_lr:.8f}), no more decay")

            # Early Stopping
            if es_counter >= es_patience:
                print(f"[Early Stop] No improvement for {es_patience} epochs. "
                      f"Best Val Loss: {best_val_loss:.6f}")
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
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
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

    args = parser.parse_args()
    main(args)
