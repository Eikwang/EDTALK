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

        # 获取头部区域比例 (默认 0.6 = 图像高度的 60%)
        self.head_ratio = getattr(args, 'head_ratio', 0.6)

        # 是否使用 landmark 检测 (默认 False，使用固定矩形)
        self.use_landmark = getattr(args, 'use_landmark', False)

        # 初始化 landmark 检测器 (如果启用)
        self._detector = None
        self._predictor = None
        if self.use_landmark:
            self._init_landmark_detector()

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
                betas=(0.5, 0.99)
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
                betas=(0.5, 0.99)
            )
        else:
            self.g_optim = optim.Adam(
                self.gen.parameters(),
                lr=args.lr * g_reg_ratio,
                betas=(0.5, 0.99)
            )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0.5, 0.99)
        )

        self.criterion_vgg = VGGLoss().to(device)

    def _init_landmark_detector(self):
        """初始化 dlib landmark 检测器"""
        try:
            import dlib
            import cv2
            import numpy as np

            # 查找 landmark 模型路径
            predictor_path = None
            candidates = [
                os.path.join(os.path.dirname(__file__), '..', 'ckpts', 'shape_predictor_68_face_landmarks.dat'),
                'shape_predictor_68_face_landmarks.dat',
            ]
            for p in candidates:
                if os.path.exists(p):
                    predictor_path = p
                    break

            if predictor_path is None:
                print("[WARN] Landmark model not found, falling back to fixed rectangle mode")
                self.use_landmark = False
                return

            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(predictor_path)
            print(f"[INFO] Landmark detector initialized from {predictor_path}")

        except ImportError as e:
            print(f"[WARN] dlib not available ({e}), falling back to fixed rectangle mode")
            self.use_landmark = False
        except Exception as e:
            print(f"[WARN] Failed to init landmark detector: {e}, falling back to fixed rectangle mode")
            self.use_landmark = False

    def _compute_head_region_from_landmark(self, img_target):
        """从 landmark 检测生成头部区域掩码

        Args:
            img_target: [B, 3, 256, 256] 目标图像

        Returns:
            head_mask: [B, 1, 256, 256] 头部区域 (1=头部, 0=背景)
        """
        import cv2
        import numpy as np

        B, C, H, W = img_target.shape
        device = img_target.device

        # 将 tensor 转换为 numpy 图像
        images_np = []
        for i in range(B):
            img = img_target[i].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = (img + 1.0) * 127.5  # [-1, 1] -> [0, 255]
            img = img.astype(np.uint8)
            images_np.append(img)

        head_masks = []

        for img_np in images_np:
            # 检测人脸
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            rects = self._detector(gray, 0)

            if len(rects) == 0:
                # 未检测到人脸，使用固定矩形
                head_start = int(H * (1.0 - self.head_ratio))
                center_x = W // 2
                half_width = int(W * 0.4)
                mask = np.zeros((H, W), dtype=np.float32)
                mask[head_start:H, center_x - half_width:center_x + half_width] = 1.0
                head_masks.append(mask)
                continue

            # 取第一个检测到的人脸
            rect = rects[0]
            shape = self._predictor(gray, rect)

            # 转换为 numpy 数组 (68个点)
            points = shape.parts()
            landmarks = np.array([(p.x, p.y) for p in points])

            # 头部区域: 使用所有关键点计算边界框
            # 点 0-16: 下巴轮廓 (最下方)
            # 点 17-26: 眉毛 (最上方)
            all_points = landmarks

            # 计算头部边界框
            head_x_min = int(all_points[:, 0].min())
            head_x_max = int(all_points[:, 0].max())
            head_y_min = int(all_points[:, 1].min())
            head_y_max = int(all_points[:, 1].max())

            # 扩展边界框 (上下各扩展 20%, 左右各扩展 15%)
            head_height = head_y_max - head_y_min
            head_width = head_x_max - head_x_min

            expand_y = int(head_height * 0.2)
            expand_x = int(head_width * 0.15)

            head_y_min = max(0, head_y_min - expand_y)
            head_y_max = min(H, head_y_max + expand_y)
            head_x_min = max(0, head_x_min - expand_x)
            head_x_max = min(W, head_x_max + expand_x)

            # 创建掩码
            mask = np.zeros((H, W), dtype=np.float32)
            mask[head_y_min:head_y_max, head_x_min:head_x_max] = 1.0

            head_masks.append(mask)

        # 转换为 tensor
        head_masks = np.stack(head_masks, axis=0)  # [B, H, W]
        head_masks = torch.from_numpy(head_masks).unsqueeze(1).to(device)  # [B, 1, H, W]

        return head_masks

    def _compute_head_region(self, img_target):
        """生成头部区域掩码

        Args:
            img_target: [B, 3, 256, 256] 目标图像

        Returns:
            head_mask: [B, 1, 256, 256] 头部区域 (1=头部, 0=背景)
        """
        if self.use_landmark and self._predictor is not None:
            return self._compute_head_region_from_landmark(img_target)

        # 固定矩形方案 (fallback)
        B, C, H, W = img_target.shape
        device = img_target.device

        # 头部区域: 图像底部中央的矩形
        # head_ratio=0.6 表示头部占图像高度的 60%
        head_start = int(H * (1.0 - self.head_ratio))
        head_end = H

        # 宽度方向: 中央区域 (约 80% 宽度)
        center_x = W // 2
        half_width = int(W * 0.4)

        # 创建掩码
        head_mask = torch.zeros(B, 1, H, W, device=device)
        head_mask[:, :, head_start:head_end, center_x - half_width:center_x + half_width] = 1.0

        return head_mask

    def _mask_regularization(self, masks, head_region):
        """计算 mask 正则化损失

        目标: 非头部区域的 mask -> 1 (纯用 source feature, 背景保持不变)

        Args:
            masks: list of [B, 1, H, W] 各分辨率层的 mask
            head_region: [B, 1, 256, 256] 头部区域掩码

        Returns:
            reg_loss: scalar 正则化损失
        """
        reg_loss = 0.0

        for mask in masks:
            # 将头部区域缩放到当前 mask 的分辨率
            h, w = mask.shape[2], mask.shape[3]
            head_resized = F.interpolate(
                head_region,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

            # 背景区域掩码 (1 - 头部区域)
            bg_mask = 1.0 - head_resized

            # 约束: 背景区域的 mask 应该接近 1
            # loss = |1 - mask| * bg_mask
            reg_loss += ((1.0 - mask) * bg_mask).abs().mean()

        return reg_loss

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

        # 主重建
        img_target_recon = gen_fwd(img_source, img_target)
        img_recon_pred = dis_fwd(img_target_recon)

        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        l1_loss = F.l1_loss(img_target_recon, img_target)
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        g_loss = vgg_loss + 100.0 * l1_loss + 0.5 * gan_g_loss

        # Mask 正则化 (如果启用)
        mask_reg_weight = getattr(self.args, 'mask_reg_weight', 0.0)
        if mask_reg_weight > 0:
            # 获取头部区域掩码
            head_region = self._compute_head_region(img_target)

            # 重新前向获取 masks (不使用 DDP wrapper, 因为需要 return_masks)
            _, masks = self.gen(img_source, img_target, return_masks=True)

            # 计算 mask 正则化损失
            mask_reg_loss = self._mask_regularization(masks, head_region)

            g_loss = g_loss + mask_reg_loss * mask_reg_weight

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
