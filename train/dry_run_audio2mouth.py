# -*- coding: utf-8 -*-
"""
Audio2Mouth Trainer 初始化 dry-run
验证: audio_encoder 分层解冻、优化器参数分组、VGG 稀疏默认值
用法: D:\\AI\\EDTalk\\runtime312\\python.exe train/dry_run_audio2mouth.py
"""
import os
import sys
import argparse
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train.trainer_audio2mouth import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=1)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_lip", type=int, default=20)
    parser.add_argument("--latent_dim_pose", type=int, default=6)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./Audio2Lip/')
    parser.add_argument("--exp_name", type=str, default='image_use_sync')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--semantic_radius", type=int, default=13)
    parser.add_argument("--log_iter", type=int, default=1000)
    parser.add_argument("--image_save_iter", type=int, default=6000)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--eval_iter", type=int, default=4000)
    parser.add_argument("--sync_weight", type=float, default=0.0)  # dry-run 关闭 sync
    parser.add_argument("--syncnet_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dis_weight", type=int, default=0)
    parser.add_argument("--vis_num", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--audio2lip_ckpt", type=str, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--data_path", type=str, default='HDTF')
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--preload_features", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vgg_sparse_freq", type=int, default=30)
    parser.add_argument("--vgg_weight", type=float, default=0.15)
    parser.add_argument("--vgg_sparse_compensate", action='store_true')
    parser.add_argument("--disable_vgg", action='store_true')
    parser.add_argument("--mouth_vgg_weight", type=float, default=0.1)
    parser.add_argument("--mouth_vgg_crop_top", type=float, default=0.4)
    parser.add_argument("--smooth_weight", type=float, default=0.2)
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--lr_schedule", type=str, default=None)
    parser.add_argument("--sync_schedule", type=str, default=None)
    # 新增分层解冻参数
    parser.add_argument("--audio_encoder_unfreeze_from", type=int, default=9)
    parser.add_argument("--audio_encoder_lr_mult", type=float, default=0.1)
    parser.add_argument("--mapping_lr_mult", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[DRY-RUN] device={device}')

    trainer = Trainer(args, device)

    # 统计可训练参数
    ae_total = sum(p.numel() for p in trainer.audio2lip.audio_encoder.parameters())
    ae_train = sum(p.numel() for p in trainer.audio2lip.audio_encoder.parameters() if p.requires_grad)
    mapping_total = sum(p.numel() for p in trainer.audio2lip.mapping.parameters())
    mapping_train = sum(p.numel() for p in trainer.audio2lip.mapping.parameters() if p.requires_grad)
    gen_total = sum(p.numel() for p in trainer.gen.parameters())
    gen_train = sum(p.numel() for p in trainer.gen.parameters() if p.requires_grad)

    print('\n=== 参数解冻统计 ===')
    print(f'audio_encoder: {ae_train}/{ae_total} 可训练 ({ae_train / ae_total * 100:.1f}%)')
    print(f'mapping      : {mapping_train}/{mapping_total} 可训练')
    print(f'generator    : {gen_train}/{gen_total} 可训练 ({gen_train / gen_total * 100:.1f}%)')

    # 验证优化器参数分组
    print('\n=== 优化器参数分组 ===')
    for i, g in enumerate(trainer.g_optim.param_groups):
        params_cnt = sum(p.numel() for p in g['params'])
        print(f'group {i}: name={g.get("name", "unknown")}, lr={g["lr"]:.6f}, params={params_cnt}')

    # 验证 VGG 稀疏配置
    print('\n=== VGG 稀疏配置 ===')
    print(f'vgg_sparse_freq={trainer.vgg_sparse_freq}, vgg_weight={trainer.vgg_weight}, '
          f'compensate={trainer.vgg_sparse_compensate}, disable_vgg={trainer.disable_vgg}')
    print(f'mouth_vgg_weight={trainer.mouth_vgg_weight}, crop_top={trainer.mouth_vgg_crop_top}, '
          f'smooth_weight={trainer.smooth_weight}')

    # 模拟一次前向+反向，验证参数分组和梯度是否正常
    print('\n=== 模拟一次训练 step ===')
    B, T = 1, 5
    audio_features = torch.randn(B, T, 80, 16, device=device)
    lip_features = torch.randn(B, T, 20, device=device)
    pose_features = torch.randn(B, T, 6, device=device)
    identity_img = torch.randn(B, 3, 256, 256, device=device)
    target_img = torch.randn(B, T, 3, 256, 256, device=device)
    bbox = torch.tensor([[0, 0, 256, 256]] * (B * T), dtype=torch.float32, device=device)

    trainer.gen.zero_grad()
    trainer.audio2lip.zero_grad()
    G_losses, recon, g_loss = trainer.gen_update(
        audio_features, lip_features, pose_features, identity_img, target_img, bbox,
        scaler=None, step_optimizer=True
    )
    print(f'g_loss={g_loss.item():.4f}, losses={ {k: v.item() for k, v in G_losses.items()} }')

    # 检查部分解冻层是否有梯度
    print('\n=== 梯度检查 ===')
    ae_blocks = list(trainer.audio2lip.audio_encoder.children())
    for idx in [0, 8, 9, 12]:
        has_grad = any(p.grad is not None for p in ae_blocks[idx].parameters())
        print(f'audio_encoder block {idx} has_grad={has_grad}')

    print('\n[DRY-RUN] OK')


if __name__ == '__main__':
    main()
