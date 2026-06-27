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
        if args.distributed:
            self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
            self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)
            self.gen = self.gen.module
            self.dis = self.dis.module

        # NOTE: 原配置 betas=(0**reg_ratio, 0.99**reg_ratio) => beta1=0(无动量)，
        # 这是 StyleGAN2 配合 path length regularization 用的。但本训练无 path length reg，
        # 且主要损失是 VGG+L1 回归任务，无动量收敛慢且抖。改为 (0.5, 0.99)。
        # lr 不再乘 reg_ratio（无 path length reg 时不适用该缩放）。
        betas = (0.5, 0.99)

        if args.only_fine_tune_dec:
            requires_grad(self.gen, False)
            requires_grad(self.gen.dec, True)
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

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr,
            betas=betas
        )

        self.criterion_vgg = VGGLoss().to(device)

        # LR warmup + cosine 衰减调度器
        # warmup 从 0 线性升到 args.lr (前 warmup_iters 步)，之后 cosine 衰减到 0
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

    def r1_reg(self, real_img):
        """R1 梯度惩罚 (StyleGAN2)。对真实图像输入求判别器输出的梯度，
        惩罚梯度的平方和，防止判别器在小数据集上过拟合导致 GAN 信号失效。
        每隔 d_reg_every 步执行一次 (lazy regularization)。
        """
        real_img.requires_grad = True
        real_pred = self.dis(real_img)
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )[0]
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def gen_update(self, img_source, img_target):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        img_target_recon = self.gen(img_source, img_target)
        img_recon_pred = self.dis(img_target_recon)

        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        l1_loss = F.l1_loss(img_target_recon, img_target)
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        # 身份保持损失：约束重建图的身份 latent 与 source 一致。
        # 旧的实现用像素 L1(recon, source) 与重建损失 L1(recon, target) 数学冲突，
        # 会让模型在两个不同目标间妥协，导致模糊。改为在 latent 空间约束身份：
        # 只约束"五官结构/脸型"等身份信息，允许姿态/表情/口型自由变化。
        # Encoder 冻结时(only_fine_tune_dec)，梯度仍可通过冻结的计算图回传到 Decoder，
        # 督促 Decoder 生成身份一致的图像。
        id_weight = getattr(self.args, 'id_weight', 0.05)
        if id_weight > 0:
            with torch.no_grad():
                wa_source, _, _, _ = self.gen.enc(img_source, None)
            wa_recon, _, _, _ = self.gen.enc(img_target_recon, None)
            id_loss = F.mse_loss(wa_recon, wa_source) * id_weight
        else:
            id_loss = 0.0

        g_loss = vgg_loss + l1_loss + gan_g_loss + id_loss

        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, id_loss, img_target_recon

    def dis_update(self, img_real, img_recon, do_r1=False):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

        # R1 正则 (lazy regularization): 每 d_reg_every 步执行一次
        # 权重按 d_reg_every 缩放 (StyleGAN2 惯例: lazy reg 乘以 reg_every/2, 平均到每步)
        if do_r1:
            r1_penalty = self.r1_reg(img_real)
            (d_loss + r1_penalty * (self.args.d_reg_every / 2.0)).backward()
        else:
            r1_penalty = torch.tensor(0.0, device=img_real.device)
            d_loss.backward()

        self.d_optim.step()

        return d_loss, r1_penalty

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

        id_weight = getattr(self.args, 'id_weight', 0.05)
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
        self.d_optim.load_state_dict(ckpt["d_optim"])

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
