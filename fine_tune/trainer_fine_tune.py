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

        if args.only_fine_tune_dec:
            requires_grad(self.gen, False)
            requires_grad(self.gen.dec, True)
            net_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())
            self.g_optim = optim.Adam(
                net_parameters,
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
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
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
            )
        else:
            self.g_optim = optim.Adam(
                self.gen.parameters(),
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
            )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        self.criterion_vgg = VGGLoss().to(device)

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, img_source, img_target):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        gen_fwd = self.gen_ddp if getattr(self.args, 'distributed', False) else self.gen
        dis_fwd = self.dis_ddp if getattr(self.args, 'distributed', False) else self.dis

        img_target_recon = gen_fwd(img_source, img_target)
        img_recon_pred = dis_fwd(img_target_recon)

        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        l1_loss = F.l1_loss(img_target_recon, img_target)
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        g_loss = vgg_loss + l1_loss + gan_g_loss
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
            img_recon = self.gen(img_source, img_target)
        return img_recon

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
