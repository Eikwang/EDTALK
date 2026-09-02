import torch
from networks.discriminator import Discriminator
from networks.generator_lip_pose import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from fine_tune.vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP


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

        # DDP: 保留 wrapper 用于 forward (梯度 all-reduce),
        # self.gen/self.dis 保留原始 module 用于 state_dict/save/load
        if args.distributed:
            self.gen_ddp = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
            self.dis_ddp = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        # 优化器配置: 与备份代码一致
        # g_reg_every/(g_reg_every+1) 缩放 lr 和 betas, 无 path length reg 时
        # beta1=0 (无动量), 适合微调预训练模型
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        # beta1: 可通过 --beta1 命令行参数配置
        # 原始训练使用 beta1=0 (无动量), 适合微调预训练模型
        # 建议微调时用 0 或 0.1, 避免过大动量覆盖预训练权重
        _beta1 = getattr(args, 'beta1', None)
        beta1 = _beta1 if _beta1 is not None else 0.0

        if args.only_fine_tune_dec:
            requires_grad(self.gen, False)
            requires_grad(self.gen.dec, True)
            net_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())
            self.g_optim = optim.Adam(
                net_parameters,
                lr=args.lr * g_reg_ratio,
                betas=(beta1, 0.99)
            )
        elif getattr(args, 'freeze_direction', False):
            # 冻结 Direction+fc+lip_fc+pose_fc, 训练 Encoder+Decoder
            requires_grad(self.gen, False)
            requires_grad(self.gen.enc, True)
            requires_grad(self.gen.dec, True)
            net_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())
            self.g_optim = optim.Adam(
                net_parameters,
                lr=args.lr * g_reg_ratio,
                betas=(beta1, 0.99)
            )
        else:
            # 分层学习率: Direction 使用更高学习率保持解耦能力
            direction_lr_ratio = getattr(args, 'direction_lr_ratio', 5.0)
            direction_params = list(self.gen.direction_lipnonlip.parameters())
            direction_param_ids = set(id(p) for p in direction_params)
            other_params = [p for p in self.gen.parameters() if id(p) not in direction_param_ids]

            self.g_optim = optim.Adam([
                {'params': direction_params, 'lr': args.lr * g_reg_ratio * direction_lr_ratio},
                {'params': other_params, 'lr': args.lr * g_reg_ratio}
            ], betas=(beta1, 0.99))

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(beta1, 0.99)
        )

        self.criterion_vgg = VGGLoss().to(device)

        # 记录冻结配置: 在 __init__ 结束时保存正确的 requires_grad 状态
        # gen_update/dis_update 会互相切换, 需要在每次 gen_update 开始时恢复
        self._save_gen_grad_state()

    def _save_gen_grad_state(self):
        """保存 Generator 各参数的 requires_grad 状态"""
        self._gen_grad_state = {name: p.requires_grad for name, p in self.gen.named_parameters()}

    def _restore_gen_grad_state(self):
        """恢复 Generator 各参数的 requires_grad 状态"""
        for name, p in self.gen.named_parameters():
            if name in self._gen_grad_state:
                p.requires_grad = self._gen_grad_state[name]

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, img_source, img_target):
        self.gen.train()
        self.gen.zero_grad()

        # 恢复 __init__ 中设置的冻结状态 (而不是无条件解冻所有 gen 参数)
        # gen_update/dis_update 会互相切换 requires_grad, 所以需要恢复正确的冻结配置
        self._restore_gen_grad_state()
        requires_grad(self.dis, False)

        gen_fwd = self.gen_ddp if getattr(self.args, 'distributed', False) else self.gen
        dis_fwd = self.dis_ddp if getattr(self.args, 'distributed', False) else self.dis

        # ---- 手动调用 encoder/decoder (而非 gen.forward()) 以便获取中间结果 ----
        # 1. 编码 source 和 target
        wa_s, wa_t, feats_s, feats_t = gen_fwd.enc(img_source, img_target)

        # 2. 从 target 提取运动方向 (lip + pose)
        shared_fc = gen_fwd.fc(wa_t)
        alpha_lip = gen_fwd.lip_fc(shared_fc)
        alpha_pose = gen_fwd.pose_fc(shared_fc)
        alpha = torch.cat([alpha_lip, alpha_pose], dim=-1)
        directions = gen_fwd.direction_lipnonlip(alpha)

        # 3. 主任务: source身份 + target运动 → 重建target图像
        latent = wa_s + directions
        img_target_recon = gen_fwd.dec(latent, None, feats_s)

        # 4. 自重建正则化: target身份 + target运动 → 重建target图像
        #    作用: 锚定Decoder，防止跨身份训练导致Decoder漂移
        #    复用已计算的 encoder 结果，仅额外一次 decoder 前向
        self_recon_weight = getattr(self.args, 'self_recon_weight', 0.5)
        if self_recon_weight > 0:
            latent_self = wa_t + directions
            recon_self = gen_fwd.dec(latent_self, None, feats_t)
            self_recon_loss = F.l1_loss(recon_self, img_target)
        else:
            self_recon_loss = torch.tensor(0.0, device=img_source.device)

        # ---- 计算损失 ----
        img_recon_pred = dis_fwd(img_target_recon)

        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        l1_loss = F.l1_loss(img_target_recon, img_target)
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        # 正交正则化: 约束 Direction 权重矩阵保持正交性
        # 防止微调过程中 lip/pose 方向纠缠导致背景变形
        ortho_loss = gen_fwd.orthogonality_loss()

        g_loss = vgg_loss + l1_loss + gan_g_loss + 0.01 * ortho_loss + self_recon_weight * self_recon_loss

        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, img_target_recon

    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        dis_fwd = self.dis_ddp if getattr(self.args, 'distributed', False) else self.dis

        real_img_pred = dis_fwd(img_real)
        recon_img_pred = dis_fwd(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target):
        with torch.no_grad():
            self.gen.eval()
            # 与 gen_update 保持一致的手动 encoder/decoder 调用
            wa_s, wa_t, feats_s, feats_t = self.gen.enc(img_source, img_target)
            shared_fc = self.gen.fc(wa_t)
            alpha_lip = self.gen.lip_fc(shared_fc)
            alpha_pose = self.gen.pose_fc(shared_fc)
            alpha = torch.cat([alpha_lip, alpha_pose], dim=-1)
            directions = self.gen.direction_lipnonlip(alpha)
            latent = wa_s + directions
            img_recon = self.gen.dec(latent, None, feats_s)
        return img_recon

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location=lambda storage, loc: storage, weights_only=False)
        ckpt_name = os.path.basename(resume_ckpt)
        try:
            start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            start_iter = 0

        # Generator: 必须存在
        self.gen.load_state_dict(ckpt["gen"])

        # Discriminator: 可选 — audio2mouth 训练流程不训练/不保存 Discriminator
        # 如果 checkpoint 中没有 dis, 使用随机初始化的 Discriminator (从头训练即可)
        if "dis" in ckpt:
            self.dis.load_state_dict(ckpt["dis"])
            print("[INFO] Discriminator weights loaded from checkpoint")
        else:
            print("[WARN] No 'dis' key in checkpoint — Discriminator uses random init (will train from scratch)")

        # 优化器状态: 可选 — 不同训练流程的优化器参数不同, 加载失败不影响训练
        try:
            self.g_optim.load_state_dict(ckpt["g_optim"])
        except:
            print("[INFO] g_optim not loaded (will start fresh)")
        try:
            self.d_optim.load_state_dict(ckpt["d_optim"])
        except:
            print("[INFO] d_optim not loaded (will start fresh)")

        return start_iter

    def save(self, idx, checkpoint_path):
        name = f"{int(idx):06d}.pt" if isinstance(idx, int) else f"{idx}.pt"

        # WDDM 防护: 显式将 GPU 张量拷贝到 CPU 再保存
        def _to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_cpu(v) for v in obj]
            else:
                return obj

        state = {
            "gen": _to_cpu(self.gen.state_dict()),
            "dis": _to_cpu(self.dis.state_dict()),
            "g_optim": _to_cpu(self.g_optim.state_dict()),
            "d_optim": _to_cpu(self.d_optim.state_dict()),
            "args": self.args
        }
        torch.save(state, os.path.join(checkpoint_path, name))
        del state
