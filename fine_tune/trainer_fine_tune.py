import torch
from networks.discriminator import Discriminator
from networks.generator_lip_pose import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from fine_tune.vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import math


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device, rank=0):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.rank = rank

        self.gen = Generator().to(device)
        self.dis = Discriminator().to(device)

        # distributed computing
        # 修复 BUG-3: 原代码 DDP 包装后立即 unwrap (self.gen = self.gen.module),
        #   导致 forward 绕过 DDP wrapper, 多卡训练梯度不同步。
        #   现保留 DDP wrapper (gen_ddp/dis_ddp) 用于 forward (梯度 all-reduce),
        #   self.gen/self.dis 保留原始 module 用于 state_dict/save/load。
        if args.distributed:
            self.gen_ddp = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
            self.dis_ddp = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        # NOTE: 原配置 betas=(0**reg_ratio, 0.99**reg_ratio) => beta1=0(无动量)，
        # 这是 StyleGAN2 配合 path length regularization 用的。但本训练无 path length reg，
        # 且主要损失是 VGG+L1 回归任务，无动量收敛慢且抖。改为 (0.5, 0.99)。
        # lr 不再乘 reg_ratio（无 path length reg 时不适用该缩放）。
        betas = (args.beta1, 0.99)

        if args.only_fine_tune_dec:
            requires_grad(self.gen, False)
            requires_grad(self.gen.dec, True)
            net_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())

            self.g_optim = optim.Adam(
                net_parameters,
                lr=args.lr,
                betas=betas
            )

        elif getattr(args, 'freeze_direction', False):
            # 冻结 Direction + fc + lip_fc + pose_fc，只训练 Encoder + Decoder
            # 目的: 防止 QR 正交基漂移和 latent->alpha 映射漂移，
            #       同时允许 Encoder 适配单人数据提升重建质量
            requires_grad(self.gen, False)
            requires_grad(self.gen.enc, True)
            requires_grad(self.gen.dec, True)
            # direction_lipnonlip, fc, lip_fc, pose_fc 保持冻结
            net_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())

            self.g_optim = optim.Adam(
                net_parameters,
                lr=args.lr,
                betas=betas
            )

        else:

            self.g_optim = optim.Adam(
                self.gen.parameters(),
                lr=args.lr,
                betas=betas
            )

        d_wd = getattr(args, 'd_weight_decay', 0.01)
        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr,
            betas=betas,
            weight_decay=d_wd,
        )

        self.criterion_vgg = VGGLoss().to(device)

        # LR warmup + cosine 衰减调度器
        # warmup 从 0 线性升到 args.lr (前 warmup_iters 步)，之后 cosine 衰减到 0
        # 每次启动训练 (含 resume) scheduler 基于 args.iter 重新开始:
        # resume 训练时用户指定 --iter 为本轮步数, --lr 为本轮学习率,
        # scheduler 从 step=0 重新 warmup+cosine, 不延续上一轮的衰减位置。
        warmup_iters = getattr(args, 'warmup_iters', 500)
        total_iters = args.iter

        def lr_lambda(step):
            if step < warmup_iters:
                return float(step) / float(max(1, warmup_iters))
            progress = float(step - warmup_iters) / float(max(1, total_iters - warmup_iters))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.g_scheduler = LambdaLR(self.g_optim, lr_lambda)
        self.d_scheduler = LambdaLR(self.d_optim, lr_lambda)

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def _mask_regularization(self, masks, mouth_region_ratio=0.4):
        """Mask 正则化: 强制非嘴部区域 mask 向 1 饱和。

        masks: list of [B, 1, H, W] 各层 ToFlow 的 mask 张量
        mouth_region_ratio: 嘴部区域占图像高度的底部比例 (0.4 = 下方 40%)

        原理: mask=1 表示完全使用 source feat (背景保持)，
              mask=0 表示完全使用 generated input (受 latent/direction 控制)。
              背景偏移的本质是 mask 在背景区域不饱和 (< 1)。
        """
        reg_loss = torch.tensor(0.0, device=masks[0].device)
        for mask in masks:
            B, C, H, W = mask.shape
            # 创建嘴部区域 mask: 图像底部 mouth_region_ratio 部分视为嘴部
            # 顶部 (1 - mouth_region_ratio) 部分视为背景区域
            mouth_start = int(H * (1.0 - mouth_region_ratio))
            # 背景区域 mask: 顶部 + 两侧
            bg_mask = torch.ones(1, 1, H, W, device=mask.device)
            bg_mask[:, :, mouth_start:, :] = 0.0  # 嘴部区域豁免
            # 约束: 背景区域 mask -> 1
            reg_loss += ((1.0 - mask) * bg_mask).abs().mean()
        return reg_loss / len(masks)

    def _compute_decouple_loss(self, img_source, img_target, img_cross):
        """单人场景解耦损失:
        - pose direction 应对 lip 变化不变 (cosine similarity 接近 1)
        - lip direction: 允许变化，但约束变化幅度合理 (L2)
        """
        # 编码 target 和 cross 的 latent
        _, wa_t_tgt, _, _ = self.gen.enc(img_source, img_target)
        _, wa_t_cross, _, _ = self.gen.enc(img_source, img_cross)

        # target 的 lip/pose direction
        shared_tgt = self.gen.fc(wa_t_tgt)
        lip_tgt = self.gen.lip_fc(shared_tgt)
        pose_tgt = self.gen.pose_fc(shared_tgt)
        alpha_tgt = torch.cat([lip_tgt, pose_tgt], -1)
        dir_tgt = self.gen.direction_lipnonlip.get_shared_out(alpha_tgt)
        dir_tgt_pose = self.gen.direction_lipnonlip.get_pose_latent(dir_tgt)

        # cross 的 lip/pose direction
        shared_cross = self.gen.fc(wa_t_cross)
        lip_cross = self.gen.lip_fc(shared_cross)
        pose_cross = self.gen.pose_fc(shared_cross)
        alpha_cross = torch.cat([lip_cross, pose_cross], -1)
        dir_cross = self.gen.direction_lipnonlip.get_shared_out(alpha_cross)
        dir_cross_pose = self.gen.direction_lipnonlip.get_pose_latent(dir_cross)

        # pose direction 应对 lip 变化不变
        pose_space = torch.exp(-F.cosine_similarity(dir_tgt_pose, dir_cross_pose)).sum()

        # lip direction: 允许变化，但约束变化幅度合理 (soft 约束)
        dir_tgt_lip = self.gen.direction_lipnonlip.get_lip_latent(dir_tgt)
        dir_cross_lip = self.gen.direction_lipnonlip.get_lip_latent(dir_cross)
        lip_space = (dir_tgt_lip - dir_cross_lip).pow(2).mean()

        return lip_space, pose_space

    def gen_update(self, img_source, img_target, img_cross=None):
        self.gen.train()
        self.gen.zero_grad()

        if self.args.only_fine_tune_dec:
            requires_grad(self.gen, False)
            requires_grad(self.gen.dec, True)
        elif getattr(self.args, 'freeze_direction', False):
            requires_grad(self.gen, False)
            requires_grad(self.gen.enc, True)
            requires_grad(self.gen.dec, True)
        else:
            requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        gen_fwd = self.gen_ddp if getattr(self.args, 'distributed', False) else self.gen
        dis_fwd = self.dis_ddp if getattr(self.args, 'distributed', False) else self.dis

        id_weight = getattr(self.args, 'id_weight', 0.5)
        mask_reg_weight = getattr(self.args, 'mask_reg_weight', 0.0)
        cross_weight = getattr(self.args, 'cross_weight', 0.0)
        lip_space_weight = getattr(self.args, 'lip_space_weight', 0.0)
        pose_space_weight = getattr(self.args, 'pose_space_weight', 0.0)

        cross_vgg_val = torch.tensor(0.0, device=img_source.device)
        cross_l1_val = torch.tensor(0.0, device=img_source.device)
        lip_space_val = torch.tensor(0.0, device=img_source.device)
        pose_space_val = torch.tensor(0.0, device=img_source.device)
        mask_reg_val = torch.tensor(0.0, device=img_source.device)

        need_masks = mask_reg_weight > 0
        masks = None
        if need_masks:
            img_target_recon, masks = gen_fwd(img_source, img_target, return_masks=True)
        else:
            img_target_recon = gen_fwd(img_source, img_target)

        has_valid_cross = (img_cross is not None and not torch.equal(img_cross, img_target))
        img_cross_recon = None

        if has_valid_cross and cross_weight > 0:
            img_cross_recon = gen_fwd(img_source, img_cross)

        if has_valid_cross and (lip_space_weight > 0 or pose_space_weight > 0):
            with torch.no_grad():
                _, wa_t_tgt, _, _ = self.gen.enc(img_source, img_target)
                _, wa_t_cross, _, _ = self.gen.enc(img_source, img_cross)
            shared_tgt = self.gen.fc(wa_t_tgt)
            lip_tgt = self.gen.lip_fc(shared_tgt)
            pose_tgt = self.gen.pose_fc(shared_tgt)
            alpha_tgt = torch.cat([lip_tgt, pose_tgt], -1)
            dir_tgt = self.gen.direction_lipnonlip.get_shared_out(alpha_tgt)
            dir_tgt_pose = self.gen.direction_lipnonlip.get_pose_latent(dir_tgt)
            dir_tgt_lip = self.gen.direction_lipnonlip.get_lip_latent(dir_tgt)

            shared_cross = self.gen.fc(wa_t_cross)
            lip_cross = self.gen.lip_fc(shared_cross)
            pose_cross = self.gen.pose_fc(shared_cross)
            alpha_cross = torch.cat([lip_cross, pose_cross], -1)
            dir_cross = self.gen.direction_lipnonlip.get_shared_out(alpha_cross)
            dir_cross_pose = self.gen.direction_lipnonlip.get_pose_latent(dir_cross)
            dir_cross_lip = self.gen.direction_lipnonlip.get_lip_latent(dir_cross)

            pose_space_val = torch.exp(-F.cosine_similarity(dir_tgt_pose, dir_cross_pose)).sum()
            lip_space_val = (dir_tgt_lip - dir_cross_lip).pow(2).mean()

        id_loss_val = torch.tensor(0.0, device=img_source.device)
        if id_weight > 0:
            with torch.no_grad():
                wa_source, _, _, _ = self.gen.enc(img_source, None)
            wa_recon, _, _, _ = self.gen.enc(img_target_recon, None)
            id_loss_val = F.mse_loss(wa_recon, wa_source) * id_weight

        img_recon_pred = dis_fwd(img_target_recon)
        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        l1_loss = F.l1_loss(img_target_recon, img_target)
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        total_loss = vgg_loss + l1_loss + gan_g_loss
        if id_weight > 0:
            total_loss = total_loss + id_loss_val
        if need_masks:
            mask_reg_val = self._mask_regularization(
                masks, mouth_region_ratio=getattr(self.args, 'mask_mouth_ratio', 0.4))
            total_loss = total_loss + mask_reg_val * mask_reg_weight
            del masks

        if has_valid_cross and cross_weight > 0 and img_cross_recon is not None:
            cross_vgg_val = self.criterion_vgg(img_cross_recon, img_source).mean()
            cross_l1_val = F.l1_loss(img_cross_recon, img_source)
            total_loss = total_loss + (cross_vgg_val + cross_l1_val) * cross_weight
            del img_cross_recon

        if has_valid_cross and (lip_space_weight > 0 or pose_space_weight > 0):
            if lip_space_weight > 0:
                total_loss = total_loss + lip_space_val * lip_space_weight
            if pose_space_weight > 0:
                total_loss = total_loss + pose_space_val * pose_space_weight

        total_loss.backward()
        del total_loss, img_recon_pred

        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, id_loss_val, img_target_recon, \
               cross_vgg_val, cross_l1_val, lip_space_val, pose_space_val, mask_reg_val

    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        dis_fwd = self.dis_ddp if getattr(self.args, 'distributed', False) else self.dis

        real_img_pred = dis_fwd(img_real)
        recon_img_pred = dis_fwd(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()

        del real_img_pred, recon_img_pred

        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target):
        with torch.no_grad():
            self.gen.eval()

            img_recon = self.gen(img_source, img_target)

        return img_recon

    @torch.no_grad()
    def validate(self, img_source, img_target):
        """在验证集上计算指标，用于监控过拟合和早停决策。
        返回 vgg_loss, l1_loss, id_loss（仅前向，不更新参数）。
        """
        self.gen.eval()
        img_target_recon = self.gen(img_source, img_target)

        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        l1_loss = F.l1_loss(img_target_recon, img_target)

        # 修复 BUG-7: id_weight 默认值与 gen_update (0.5) 对齐, 原为 0.0
        id_weight = getattr(self.args, 'id_weight', 0.5)
        if id_weight > 0:
            wa_source, _, _, _ = self.gen.enc(img_source, None)
            wa_recon, _, _, _ = self.gen.enc(img_target_recon, None)
            id_loss = F.mse_loss(wa_recon, wa_source) * id_weight
        else:
            id_loss = torch.tensor(0.0, device=img_source.device)

        return vgg_loss, l1_loss, id_loss

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location=lambda storage, loc: storage, weights_only=False)
        ckpt_name = os.path.basename(resume_ckpt)
        try:
            start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            start_iter = 0
        self.gen.load_state_dict(ckpt["gen"])
        self.dis.load_state_dict(ckpt["dis"])
        try:
            self.g_optim.load_state_dict(ckpt["g_optim"])
        except:
            pass
        try:
            self.d_optim.load_state_dict(ckpt["d_optim"])
        except:
            pass

        return start_iter

    def save(self, idx, checkpoint_path):
        # idx 为整数时补零 (000123.pt)，为字符串 (best/final_best) 时原样使用
        name = f"{int(idx):06d}.pt" if isinstance(idx, (int,)) else f"{idx}.pt"
        torch.save(
            {
                "gen": self.gen.state_dict(),
                "dis": self.dis.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            os.path.join(checkpoint_path, name)
        )
