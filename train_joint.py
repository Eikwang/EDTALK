#!/usr/bin/env python3
"""
联合训练脚本：Audio2Lip + Generator 端到端训练
目标：在小批量专人数据集上训练出能还原人物口型特征的模型

核心思想：
1. Audio2Lip 预测唇部特征
2. Generator 根据唇部特征生成图像
3. 端到端优化，两个模型协同学习
4. 使用更合理的损失函数

用法：
python train_joint.py --data_path wy --epoch 20 --batch_size 8 --lr 0.0002
"""

import os
import sys

# 环境变量设置（必须在最前面）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 直接导入需要的模块，避免导入复杂的 Trainer 类
from datasets.dataset_audio2lip import Audio2LipDataset_image_sync
from networks.generator_lip_pose import Generator
from networks.audio_encoder import Audio2Lip


class JointTrainer:
    """联合训练器：Audio2Lip + Generator 端到端训练"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # 初始化模型
        self.audio2lip = Audio2Lip().to(device)
        self.generator = Generator(
            args.size, 
            args.latent_dim_style, 
            args.latent_dim_lip, 
            args.latent_dim_pose, 
            args.channel_multiplier
        ).to(device)
        
        # 冻结Generator的Encoder，只训练Decoder
        # 与原始训练脚本一致
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.generator.dec.parameters():
            param.requires_grad = True
            
        # 优化器
        self.audio2lip_optimizer = torch.optim.Adam(
            self.audio2lip.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999)
        )
        self.generator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=args.lr * 0.1,  # Generator使用更小的学习率
            betas=(0.9, 0.999)
        )
        
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # 训练状态
        self.iteration = 0
        
    def train_step(self, batch, is_validation=False):
        """执行一步训练
        
        Args:
            batch: 数据批次
            is_validation: 是否为验证阶段（验证阶段不执行反向传播）
        """
        # 解包数据
        audio_features = batch['audio_features'].to(self.device)
        lip_features = batch['lip_features'].to(self.device)
        pose_features = batch['pose_features'].to(self.device)
        identity_img = batch['identity_img'].to(self.device)
        target_img = batch['target_img'].to(self.device)
        
        batch_size, T = audio_features.shape[0], audio_features.shape[1]
        
        # 前向传播
        # 1. Audio2Lip预测唇部特征（与原始训练脚本一致）
        audio_input = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
        lip_features_pred = self.audio2lip(audio_input, batch_size, T)
        
        # 2. 计算唇部特征损失
        lip_loss = self.mse_loss(lip_features_pred, lip_features)
        lip_smooth_loss = self.mse_loss(
            lip_features_pred[:, 1:] - lip_features_pred[:, :-1],
            lip_features[:, 1:] - lip_features[:, :-1]
        ) * 0.5
        
        # 3. Generator生成图像（与推理流程一致）
        # 编码源图像和目标图像（获取真实的pose信息）
        # 注意：target_img是(batch_size, T, 3, 256, 256)的5D张量，需要reshape成4D
        # 使用target_img的最后一帧作为驱动图像（与推理时的pose_img_drive对应）
        target_img_flat = target_img.reshape([batch_size * T, 3, 256, 256])
        target_img_last = target_img[:, -1, :, :, :]  # 取最后一帧，形状为(batch_size, 3, 256, 256)
        
        wa_source, wa_target, feats_source, feats_target = self.generator.enc(identity_img, target_img_last)
        
        # 使用目标图像的编码计算pose部分（与推理流程一致）
        shared_fc = self.generator.fc(wa_target)
        alpha_D_pose = self.generator.pose_fc(shared_fc)
        
        # 使用Audio2Lip预测的lip特征（与推理流程一致）
        lip_features_pred_flat = lip_features_pred.reshape([batch_size * T, 20])
        alpha_D_lip = lip_features_pred_flat
        
        # 组合lip和pose（与推理流程一致）
        # 注意：alpha_D_pose是(batch_size, 6)，需要重复T次以匹配alpha_D_lip的(batch_size*T, 20)
        alpha_D_pose_expanded = alpha_D_pose.repeat(T, 1)  # (batch_size*T, 6)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose_expanded], dim=-1)
        directions = self.generator.direction_lipnonlip(alpha_D)
        
        # 生成潜在向量（与推理流程一致）
        rep = torch.LongTensor([T] * batch_size).to(self.device)
        wa_source = torch.repeat_interleave(wa_source, rep, dim=0)
        feats_source = [torch.repeat_interleave(feat, rep, dim=0) for feat in feats_source]
        
        latent = wa_source + directions
        
        # 解码生成图像
        recon = self.generator.dec(latent, None, feats_source)
        
        # 4. 计算图像重建损失
        img_l1_loss = self.l1_loss(recon, target_img_flat) * 10
        img_smooth_loss = self.l1_loss(
            recon[:, :, :, 1:] - recon[:, :, :, :-1],
            target_img_flat[:, :, :, 1:] - target_img_flat[:, :, :, :-1]
        ) * 0.5
        
        # 5. 计算嘴部区域损失（简化版）
        # 使用固定区域裁剪嘴部
        mouth_region_recon = recon[:, :, 128:, :]  # 下半部分
        mouth_region_target = target_img_flat[:, :, 128:, :]
        mouth_l1_loss = self.l1_loss(mouth_region_recon, mouth_region_target) * 5
        
        # 6. 正交正则化（与fine_tune训练一致）
        ortho_loss = self.generator.orthogonality_loss()
        
        # 7. 总损失
        total_loss = (
            lip_loss * 10 +           # 唇部特征损失
            lip_smooth_loss * 5 +      # 唇部平滑损失
            img_l1_loss +              # 图像重建损失
            img_smooth_loss +          # 图像平滑损失
            mouth_l1_loss +            # 嘴部区域损失
            ortho_loss * 0.01          # 正交正则化
        )
        
        # 反向传播（仅在训练阶段执行）
        if not is_validation:
            self.audio2lip_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            total_loss.backward()
            self.audio2lip_optimizer.step()
            self.generator_optimizer.step()
            
            self.iteration += 1
        
        # 返回损失字典
        losses = {
            'total_loss': total_loss.item(),
            'lip_loss': lip_loss.item(),
            'lip_smooth_loss': lip_smooth_loss.item(),
            'img_l1_loss': img_l1_loss.item(),
            'img_smooth_loss': img_smooth_loss.item(),
            'mouth_l1_loss': mouth_l1_loss.item()
        }
        
        return losses, recon, lip_features_pred
    
    def save(self, checkpoint_path, epoch, iteration):
        """保存检查点"""
        checkpoint = {
            'audio2lip': self.audio2lip.state_dict(),
            'generator': self.generator.state_dict(),
            'audio2lip_optimizer': self.audio2lip_optimizer.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'epoch': epoch,
            'iteration': iteration
        }
        torch.save(checkpoint, os.path.join(checkpoint_path, f'joint_epoch_{epoch:04d}_iter_{iteration:06d}.pt'))
        
    def load(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio2lip.load_state_dict(checkpoint['audio2lip'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.audio2lip_optimizer.load_state_dict(checkpoint['audio2lip_optimizer'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        return checkpoint['epoch'], checkpoint['iteration']


def main(args):
    """主训练函数"""
    # CUDA预热
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"[INFO] CUDA 预热中... GPU: {torch.cuda.get_device_name(0)}")
        _dummy = torch.empty(1, device=device)
        del _dummy
        torch.cuda.empty_cache()
        print(f"[INFO] CUDA 预热完成")
    
    # 创建日志目录
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "").replace(":", "").replace(" ", "_")
    log_path = os.path.join('logs', 'joint_training', args.exp_name, date, 'log')
    checkpoint_path = os.path.join('logs', 'joint_training', args.exp_name, date, 'checkpoint')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    # 准备数据集
    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    dataset = Audio2LipDataset_image_sync(
        hdtf=args.data_path, 
        is_train=True, 
        transform=transform,
        preload_features=args.preload_features
    )
    
    # 测试集
    dataset_test = Audio2LipDataset_image_sync(
        hdtf=args.data_path, 
        is_train=False, 
        transform=transform,
        preload_features=args.preload_features
    )
    test_sample_count = min(200, len(dataset.samples))
    random.seed(42)
    dataset_test.samples = random.sample(dataset.samples, test_sample_count)
    random.seed()
    
    print(f"[Dataset] 训练集: {len(dataset.samples)} 样本, 测试集: {len(dataset_test.samples)} 样本")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0, 
        pin_memory=False, 
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0, 
        pin_memory=False, 
        drop_last=True
    )
    
    trainSteps = len(dataloader)
    print(f'[INFO] 训练步数: {trainSteps} / epoch')
    
    # 初始化训练器
    print('==> initializing joint trainer')
    trainer = JointTrainer(args, device)
    
    # 恢复检查点
    start_epoch = 0
    if args.resume_ckpt:
        start_epoch, _ = trainer.load(args.resume_ckpt)
        print(f'==> resume from epoch {start_epoch}')
    
    # 开始训练
    print('==> training')
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epoch):
        trainer.audio2lip.train()
        trainer.generator.train()
        
        epoch_losses = {}
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 训练一步
            losses, recon, lip_pred = trainer.train_step(batch)
            
            # 累积损失
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + value
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses["total_loss"]:.4f}',
                'lip': f'{losses["lip_loss"]:.4f}',
                'img': f'{losses["img_l1_loss"]:.4f}'
            })
            
            # 写入TensorBoard
            if batch_idx % 100 == 0:
                for key, value in losses.items():
                    writer.add_scalar(f'train/{key}', value, trainer.iteration)
        
        # 计算平均损失
        avg_losses = {key: value / trainSteps for key, value in epoch_losses.items()}
        print(f'\nEpoch {epoch+1} 平均损失:')
        for key, value in avg_losses.items():
            print(f'  {key}: {value:.4f}')
        
        # 验证
        trainer.audio2lip.eval()
        trainer.generator.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                losses, _, _ = trainer.train_step(batch, is_validation=True)
                val_loss += losses['total_loss']
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        print(f'  验证损失: {avg_val_loss:.4f}')
        
        # 写入验证损失
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            trainer.save(checkpoint_path, epoch, trainer.iteration)
            print(f'  保存最佳模型 (loss: {best_loss:.4f})')
        
        # 保存最新模型（每个epoch结束都保存，防止训练崩溃丢失进度）
        trainer.save(checkpoint_path, epoch, trainer.iteration)
        print(f'  保存epoch {epoch+1} 模型')
    
    writer.close()
    print('==> training complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Training: Audio2Lip + Generator')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='wy', help='数据集路径')
    parser.add_argument('--size', type=int, default=256, help='图像大小')
    parser.add_argument('--preload_features', action='store_true', help='预加载特征到内存')
    
    # 模型参数
    parser.add_argument('--latent_dim_style', type=int, default=512, help='风格潜在维度')
    parser.add_argument('--latent_dim_lip', type=int, default=20, help='唇部潜在维度')
    parser.add_argument('--latent_dim_pose', type=int, default=6, help='姿态潜在维度')
    parser.add_argument('--channel_multiplier', type=int, default=1, help='通道倍数')
    
    # 训练参数
    parser.add_argument('--epoch', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--save_freq', type=int, default=5, help='保存频率')
    parser.add_argument('--exp_name', type=str, default='joint_v1', help='实验名称')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='恢复检查点路径')
    
    args = parser.parse_args()
    main(args)
