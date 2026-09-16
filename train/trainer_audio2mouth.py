import torch

import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from collections import OrderedDict
from networks_Lip_NonLip.discriminator import Discriminator
from networks_Lip_NonLip.generator import Generator

from networks_audio2lip.bilinear import crop_bbox_batch
from networks_audio2lip.syncnet import SyncNet_color as SyncNet
from networks_audio2lip.audio_encoder import Audio2Lip
import numpy as np

def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.dis_weight = args.dis_weight
        self.audio2lip = Audio2Lip().to(device)
        self.train_generator = True

        # ============================================================
        # audio_encoder 分层解冻
        # ============================================================
        # audio_encoder 为 nn.Sequential, 共 13 个 Conv2d block:
        #   blocks 0-2 : 32 通道浅层, 学习低级谱特征 (与 speaker 无关, 保持冻结)
        #   blocks 3-5 : 64 通道, 过渡层
        #   blocks 6-8 : 128 通道
        #   blocks 9   : 256 通道下采样
        #   blocks 10  : 256 通道残差
        #   blocks 11-12: 512 通道深层, 靠近语义输出
        #   mapping    : Linear(512->20), 必须训练以适配目标 speaker
        #
        # --audio_encoder_unfreeze_from=N: 从第 N 个 block 开始解冻 (0=全部解冻, 13=只解冻 mapping)
        # 默认 9: 只解冻深层 (blocks 9-12) + mapping, 保留浅层 Wav2Lip 预训练特征
        # ============================================================
        audio_encoder_blocks = list(self.audio2lip.audio_encoder.children())
        unfreeze_from = getattr(args, 'audio_encoder_unfreeze_from', 9)
        unfreeze_from = max(0, min(unfreeze_from, len(audio_encoder_blocks)))

        for i, block in enumerate(audio_encoder_blocks):
            freeze = (i < unfreeze_from)
            requires_grad(block, not freeze)
            if not freeze:
                print(f'[audio_encoder] block {i} unfrozen ({type(block).__name__})')

        # mapping 层始终训练: 把 512 维音频特征映射到 20 维 lip features
        requires_grad(self.audio2lip.mapping, True)
        self.audio_encoder_unfreeze_from = unfreeze_from
        self.audio_encoder_lr_mult = getattr(args, 'audio_encoder_lr_mult', 0.1)
        self.mapping_lr_mult = getattr(args, 'mapping_lr_mult', 1.0)

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_lip, args.latent_dim_pose, args.channel_multiplier).to(
            device)
        requires_grad(self.gen, False)
        requires_grad(self.gen.dec, True)
        if self.dis_weight !=0:
            self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        # requires_grad(self.gen.dec, False)
        # requires_grad(self.gen.fc, False)
        if args.distributed:
            self.audio2lip = nn.parallel.DistributedDataParallel(
                self.audio2lip,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
                )
            if self.train_generator:
                self.gen = nn.parallel.DistributedDataParallel(
                    self.gen,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                    )
            # self.dis = nn.parallel.DistributedDataParallel(
            #     self.dis,
            #     device_ids=[args.local_rank],
            #     output_device=args.local_rank,
            #     broadcast_buffers=False,
            #     find_unused_parameters=True,
            #     )
        # distributed computing
        # self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        # self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        # d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        if args.distributed:
            self.audio2lip = self.audio2lip.module
            self.gen = self.gen.module


        # ============================================================
        # 优化器参数分组: 为 audio_encoder 深层、mapping、generator 分别设置学习率
        # ============================================================
        base_lr = args.lr * g_reg_ratio

        # audio_encoder 深层参数
        ae_unfrozen_params = []
        for i, block in enumerate(audio_encoder_blocks):
            if i >= unfreeze_from:
                ae_unfrozen_params.extend(block.parameters())

        # mapping 层参数
        mapping_params = list(self.audio2lip.mapping.parameters())

        # generator 可训练参数 (主要是 gen.dec)
        gen_params = list(filter(lambda p: p.requires_grad, self.gen.parameters()))

        param_groups = []
        if ae_unfrozen_params:
            param_groups.append({
                'params': ae_unfrozen_params,
                'lr': base_lr * self.audio_encoder_lr_mult,
                'name': 'audio_encoder',
            })
        param_groups.append({
            'params': mapping_params,
            'lr': base_lr * self.mapping_lr_mult,
            'name': 'mapping',
        })
        if gen_params:
            param_groups.append({
                'params': gen_params,
                'lr': base_lr,
                'name': 'generator',
            })

        # 修正: 原 betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio) 导致 beta1=0.0,
        # Adam 无一阶动量退化为类 RMSProp, 训练不稳定。
        # 改为标准 Adam betas (0.9, 0.999), lr 已通过 g_reg_ratio 缩放。
        self.g_optim = optim.Adam(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.999)
        )
        print(f'[Optimizer] {len(param_groups)} param groups | '
              f'audio_encoder lr={base_lr * self.audio_encoder_lr_mult:.6f} '
              f'(from block {unfreeze_from}), mapping lr={base_lr * self.mapping_lr_mult:.6f}, '
              f'generator lr={base_lr:.6f}')



        # self.d_optim = optim.Adam(
        #     self.dis.parameters(),
        #     lr=args.lr * d_reg_ratio,
        #     betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        # )

        # self.criterion_vgg = VGGLoss().to('cuda')

        self.criterion_vgg = VGGLoss().to('cuda')

        # VGG 稀疏计算：每 vgg_sparse_freq 个 iter 计算一次 VGG
        # 注意：默认不再做权重 ×N 补偿。经验上补偿会放大单步梯度，导致模型崩坏；
        #      关闭补偿后 VGG 总贡献下降，但单步梯度更稳定。
        #      如需恢复旧行为，可传 --vgg_sparse_compensate。
        self.vgg_sparse_freq = getattr(args, 'vgg_sparse_freq', 30)
        self.vgg_weight = getattr(args, 'vgg_weight', 0.15)
        self.vgg_sparse_compensate = getattr(args, 'vgg_sparse_compensate', False)
        self.disable_vgg = getattr(args, 'disable_vgg', False)
        self._train_iter = 0

        # 嘴部 crop VGG：对脸部 96x96 crop 的下半区（嘴+下巴）每 iter 计算 VGG。
        # 背景: 全图 VGG 稀疏 50 后有效权重 ≈0.003，牙齿/口腔等高频细节的感知信号
        # 几乎为零，L1×25 主导推动均值化。96x96 crop 的 VGG 计算量约为全图 1/7，
        # 可承受每 iter 计算，把感知细节信号恢复到正常量级。
        self.mouth_vgg_weight = getattr(args, 'mouth_vgg_weight', 0.1)
        self.mouth_vgg_crop_top = getattr(args, 'mouth_vgg_crop_top', 0.4)

        # recon_smooth 权重: 惩罚预测帧间变化与真实帧间变化的偏差。
        # 原 0.5 硬编码会抑制爆破音所需的快速唇形变化（1-2 帧内完成闭口），
        # 降到 0.2 减少对快速唇动的抑制。
        self.smooth_weight = getattr(args, 'smooth_weight', 0.2)

        # 开合度维度加权 MSE：teacher forcing 诊断确认 audio2lip 预测误差集中在
        # 唇部开合度维度 (默认 2,13,16 —— 扰动渲染证实这三维控制嘴部开合)，
        # 且误差方向是向训练集统计先验收缩（该闭不闭/该张不张）。
        # 对这些维度加权让模型重点学习开合定位。
        lip_dims = getattr(args, 'lip_dims', '2,13,16')
        lip_dim_weight = getattr(args, 'lip_dim_weight', 3.0)
        self.lip_dim_weight_vec = None
        if lip_dims and lip_dim_weight > 1.0:
            dims = [int(d) for d in lip_dims.split(',')]
            w = torch.ones(20)
            for d in dims:
                if 0 <= d < 20:
                    w[d] = lip_dim_weight
            self.lip_dim_weight_vec = w.to('cuda')
            print(f'[INFO] lip 开合度维度加权: dims={dims}, weight={lip_dim_weight}')

        self.sync_weight = args.sync_weight

        if self.sync_weight != 0:
            self.syncnet = SyncNet()
            # SyncNet 预训练权重路径（可选，缺失时跳过 sync loss）
            syncnet_ckpt = getattr(args, 'syncnet_ckpt', None) or 'ckpts/EDTalk.pt'
            if os.path.exists(syncnet_ckpt):
                try:
                    syncnet_state = torch.load(syncnet_ckpt, map_location='cpu', weights_only=False)

                    # 尝试多种键名格式加载 SyncNet
                    loaded = False

                    # ==== 支持 Wav2Lip.pt (TorchScript 格式) ====
                    if isinstance(syncnet_state, torch.jit.RecursiveScriptModule):
                        print('[INFO] Detected TorchScript format (Wav2Lip), extracting SyncNet weights...')
                        try:
                            sd = syncnet_state.state_dict()
                            mapped_state = {}
                            for k, v in sd.items():
                                # face_encoder 映射
                                if k.startswith('face_encoder_blocks'):
                                    new_k = k.replace('face_encoder_blocks', 'face_encoder')
                                    mapped_state[new_k] = v
                                # audio_encoder: Wav2Lip 和 SyncNet 结构不同，
                                # 仅加载形状匹配的层，跳过不匹配的层
                                elif k.startswith('audio_encoder'):
                                    # 检查当前 SyncNet 中是否存在且形状匹配的键
                                    if k in self.syncnet.state_dict():
                                        if v.shape == self.syncnet.state_dict()[k].shape:
                                            mapped_state[k] = v
                                        else:
                                            print(f'[WARN] Shape mismatch for {k}: checkpoint {v.shape} vs model {self.syncnet.state_dict()[k].shape}, skipping')
                            self.syncnet.load_state_dict(mapped_state, strict=False)
                            matched = len(mapped_state)
                            total = len(self.syncnet.state_dict())
                            print(f'[INFO] Loaded {matched}/{total} SyncNet keys from Wav2Lip.pt')
                            loaded = True
                        except Exception as e:
                            print(f'[WARN] Failed to load from Wav2Lip.pt: {e}')
                    # ==== 结束新增 ====

                    if not loaded:
                        for key in ['state_dict', 'syncnet', 'gen', 'netG']:
                            if key in syncnet_state:
                                s = syncnet_state[key]
                                try:
                                    self.syncnet.load_state_dict(s)
                                    print(f'[INFO] Loaded SyncNet from {syncnet_ckpt}[{key}]')
                                    loaded = True
                                    break
                                except:
                                    pass

                    if not loaded:
                        print(f'[WARN] {syncnet_ckpt} does not contain valid SyncNet weights, skipping')
                        self.sync_weight = 0

                except Exception as e:
                    print(f'[WARN] Failed to load SyncNet from {syncnet_ckpt}: {e}')
                    self.sync_weight = 0
            else:
                print(f'[WARN] SyncNet checkpoint not found: {syncnet_ckpt}, sync loss disabled')
                self.sync_weight = 0

            if self.sync_weight != 0:
                for p in self.syncnet.parameters():
                    p.requires_grad = False
                if torch.cuda.is_available():
                    self.syncnet = self.syncnet.cuda()
                    self.syncnet.eval()

        self.start_iter = 0

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, audio_features, lip_features, pose_features, identity_img, target_img, bbox = None, scaler=None, step_optimizer=True): # torch.Size([64, 5, 80, 16])

        self.audio2lip.train()
        self.gen.train()
        # 梯度累积: 仅在实际执行 optimizer step 时清零梯度
        # 非 step 时保留已有梯度，实现累积效果
        if step_optimizer:
            self.gen.zero_grad()
            self.audio2lip.zero_grad()
        self._train_iter += 1
        G_losses = {}

        # AMP autocast: 前向传播在 FP16 下执行，减少显存占用并加速计算
        # 注意: backward 仍需 FP32 (GradScaler 自动处理), SyncNet/crop_bbox 等
        # 自定义操作在 autocast 下安全（PyTorch 会自动选择合适精度）
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            batch_size, T = audio_features.shape[0], audio_features.shape[1]
            audio_features = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
            lip_features_predict = self.audio2lip(audio_features, batch_size, T) # batch, T, 20
            # 开合度维度加权 MSE（lip_dim_weight_vec 为 None 时退化为普通 MSE）
            if self.lip_dim_weight_vec is not None:
                diff_sq = (lip_features_predict - lip_features) ** 2
                G_losses['recon_l2_loss'] = (diff_sq * self.lip_dim_weight_vec).mean()
            else:
                G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
            G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*self.smooth_weight

            wa_identity, _, feats_identity, _ = self.gen.enc(identity_img)
            lip_features_predict = lip_features_predict.reshape([batch_size*T, 20])
            pose_features = pose_features.reshape([batch_size*T, 6])
            alpha = torch.cat([lip_features_predict, pose_features], dim=-1)
            directions = self.gen.direction_lipnonlip(alpha)
            rep = torch.LongTensor([T]*batch_size).cuda()
            wa_identity = torch.repeat_interleave(wa_identity, rep, dim=0)
            layer_num = len(feats_identity)
            for i in range(layer_num):
                feats_identity[i] = torch.repeat_interleave(feats_identity[i], rep, dim=0)

            latent = wa_identity + directions

            recon = self.gen.dec(latent, None, feats_identity)# torch.Size([20, 3, 256, 256])
            if self.dis_weight !=0:
                recon_pred = self.dis(recon)
            target_img = target_img.reshape([batch_size*T, 3, 256, 256])

            # VGG 感知损失：支持稀疏计算 + 可控权重
            # vgg_sparse_freq=0: 每 iter 计算 VGG，权重 --vgg_weight
            # vgg_sparse_freq=N: 每 N iter 计算一次 VGG，权重 --vgg_weight（默认不补偿）
            # disable_vgg=True: 完全关闭 VGG loss
            if self.disable_vgg:
                G_losses['recon_vgg_loss'] = torch.tensor(0.0, device=recon.device)
            elif self.vgg_sparse_freq > 0 and self._train_iter % self.vgg_sparse_freq != 0:
                G_losses['recon_vgg_loss'] = torch.tensor(0.0, device=recon.device)
            else:
                vgg_weight = self.vgg_weight
                if self.vgg_sparse_compensate and self.vgg_sparse_freq > 0:
                    vgg_weight *= self.vgg_sparse_freq
                G_losses['recon_vgg_loss'] = self.criterion_vgg(recon, target_img).mean() * vgg_weight

            # 嘴部 crop VGG：每 iter 计算，恢复牙齿/口腔细节的感知信号
            if self.mouth_vgg_weight > 0:
                G_losses['mouth_vgg_loss'] = self.mouth_vgg_loss(recon, target_img, bbox, batch_size, T)

            if self.dis_weight !=0:
                G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred)

            G_losses['recon_l1_loss'] = F.l1_loss(recon, target_img) * 25

            if self.sync_weight != 0:

                preds = bbox.reshape(batch_size*T, 4)
                preds = preds.to('cuda')/256.
                box_to_feat = torch.from_numpy(np.array([i for i in range(batch_size*T)]))
                gt_bbox = crop_bbox_batch(target_img, preds, box_to_feat, 96)
                pre_bbox = crop_bbox_batch(recon, preds, box_to_feat, 96)

                G_losses['img_l1_sync'] = torch.abs(gt_bbox-pre_bbox).mean() * 8
                pre_bbox = pre_bbox.reshape(batch_size, T, 3, 96, 96).permute(0, 2, 1, 3, 4)
                # 与训练端 SyncNet 输入格式一致：将 T 帧 mel 沿时间轴拼接为连续时频谱图
                # audio_features: (B*T, 1, 80, 16)
                # → reshape(B, T, 80, 16) → permute(0,2,1,3) → reshape(B, 1, 80, T*16)
                mel_for_sync = audio_features.reshape(batch_size, T, 80, 16)
                mel_for_sync = mel_for_sync.permute(0, 2, 1, 3).reshape(batch_size, 1, 80, T * 16)
                value = self.get_sync_loss(mel_for_sync, pre_bbox, 'cuda').mean()
                G_losses['sync'] = self.sync_weight * value

            G_losses_values = [val.mean() for val in G_losses.values()]
            g_loss = sum(G_losses_values)

        # AMP backward: scaler 缩放 loss 防止 FP16 梯度下溢, backward 后自动 unscale
        # 梯度累积: 每次都 backward 累积梯度, 但只在 step_optimizer 时执行 optimizer step
        if use_amp:
            scaler.scale(g_loss).backward()
            if step_optimizer:
                scaler.step(self.g_optim)
                scaler.update()
        else:
            g_loss.backward()
            if step_optimizer:
                self.g_optim.step()
        recon = recon.reshape(batch_size, T, 3,256,256)
        return G_losses, recon[:,-1], g_loss

    def get_sync_loss(self, mel, g, device):
        """SyncNet 同步损失：极小化 (1 - cos_sim)，推动唇形与音频匹配。
        与 train_syncnet.py 的 InfoNCE 训练目标一致——高 cos_sim 表示同步。"""
        g = g[:, :, :, g.size(3)//2:]
        g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=1)
        a, v = self.syncnet(mel, g)
        loss = (1 - F.cosine_similarity(a, v)).mean()
        return loss

    def mouth_vgg_loss(self, recon, target_img, bbox, batch_size, T):
        """嘴部 crop VGG：脸部 96x96 crop 的下半区（嘴+下巴）的感知损失。

        每 iter 计算，恢复被全图 VGG 稀疏化稀释的细节信号（牙齿/口腔高频纹理）。
        crop 逻辑与 sync loss 一致（crop_bbox_batch 96x96），额外取下半区聚焦嘴部。
        下半区高度约 58px，VGG19 池化 5 次会越界，先上采样到 128x128 再进 VGG。
        """
        preds = bbox.reshape(batch_size*T, 4)
        preds = preds.to('cuda')/256.
        box_to_feat = torch.from_numpy(np.array([i for i in range(batch_size*T)]))
        gt_bbox = crop_bbox_batch(target_img, preds, box_to_feat, 96)
        pre_bbox = crop_bbox_batch(recon, preds, box_to_feat, 96)
        top = int(96 * self.mouth_vgg_crop_top)
        gt_mouth = gt_bbox[:, :, top:, :]
        pre_mouth = pre_bbox[:, :, top:, :]
        gt_mouth = F.interpolate(gt_mouth, size=(128, 128), mode='bilinear', align_corners=False)
        pre_mouth = F.interpolate(pre_mouth, size=(128, 128), mode='bilinear', align_corners=False)
        return self.criterion_vgg(pre_mouth, gt_mouth).mean() * self.mouth_vgg_weight


    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, audio_features, lip_features, pose_features, identity_img, target_img, bbox):
        with torch.no_grad():
            self.audio2lip.eval()
            G_losses = {}

            batch_size, T = audio_features.shape[0], audio_features.shape[1]
            audio_features = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
            lip_features_predict = self.audio2lip(audio_features, batch_size, T) # batch, T, 20
            # 开合度维度加权 MSE（lip_dim_weight_vec 为 None 时退化为普通 MSE）
            if self.lip_dim_weight_vec is not None:
                diff_sq = (lip_features_predict - lip_features) ** 2
                G_losses['recon_l2_loss'] = (diff_sq * self.lip_dim_weight_vec).mean()
            else:
                G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
            G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*self.smooth_weight

            wa_identity, _, feats_identity, _ = self.gen.enc(identity_img)
            lip_features_predict = lip_features_predict.reshape([batch_size*T, 20])
            pose_features = pose_features.reshape([batch_size*T, 6])
            alpha = torch.cat([lip_features_predict, pose_features], dim=-1)
            directions = self.gen.direction_lipnonlip(alpha)
            rep = torch.LongTensor([T]*batch_size).cuda()
            wa_identity = torch.repeat_interleave(wa_identity, rep, dim=0)
            layer_num = len(feats_identity)
            for i in range(layer_num):
                feats_identity[i] = torch.repeat_interleave(feats_identity[i], rep, dim=0)

            latent = wa_identity + directions

            recon = self.gen.dec(latent, None, feats_identity)# torch.Size([20, 3, 256, 256])
            if self.dis_weight !=0:
                recon_pred = self.dis(recon)
            target_img = target_img.reshape([batch_size*T, 3, 256, 256])

            if self.disable_vgg:
                G_losses['recon_vgg_loss'] = torch.tensor(0.0, device=recon.device)
            else:
                G_losses['recon_vgg_loss'] = self.criterion_vgg(recon, target_img).mean() * 0.15
            # eval 与训练目标保持一致：嘴部 VGG 参与 best model 选择
            if self.mouth_vgg_weight > 0:
                G_losses['mouth_vgg_loss'] = self.mouth_vgg_loss(recon, target_img, bbox, batch_size, T)
            if self.dis_weight !=0:
                G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred)

            G_losses['recon_l1_loss'] = F.l1_loss(recon, target_img) * 25

            if self.sync_weight != 0:

                preds = bbox.reshape(batch_size*T, 4)
                preds = preds.to('cuda')/256.
                box_to_feat = torch.from_numpy(np.array([i for i in range(batch_size*T)]))
                gt_bbox = crop_bbox_batch(target_img, preds, box_to_feat, 96)
                pre_bbox = crop_bbox_batch(recon, preds, box_to_feat, 96)

                G_losses['img_l1_sync'] = torch.abs(gt_bbox-pre_bbox).mean() * 8
                pre_bbox = pre_bbox.reshape(batch_size, T, 3, 96, 96).permute(0, 2, 1, 3, 4)
                # 与训练端 SyncNet 输入格式一致：将 T 帧 mel 沿时间轴拼接为连续时频谱图
                mel_for_sync = audio_features.reshape(batch_size, T, 80, 16)
                mel_for_sync = mel_for_sync.permute(0, 2, 1, 3).reshape(batch_size, 1, 80, T * 16)
                value = self.get_sync_loss(mel_for_sync, pre_bbox, 'cuda').mean()
                G_losses['sync'] = self.sync_weight * value

            G_losses_values = [val.mean() for val in G_losses.values()]
            g_loss = sum(G_losses_values)


            recon = recon.reshape(batch_size, T, 3,256,256)
            return G_losses, recon[:,-1], g_loss

    def sample_no_loss(self, audio_features, lip_features, pose_features, identity_img, target_img):
        with torch.no_grad():
            self.audio2lip.eval()
            G_losses = {}

            batch_size, T = audio_features.shape[0], audio_features.shape[1]
            audio_features = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
            lip_features_predict = self.audio2lip(audio_features, batch_size, T) # batch, T, 20
            wa_identity, _, feats_identity, _ = self.gen.enc(identity_img)
            lip_features_predict = lip_features_predict.reshape([batch_size*T, 20])
            pose_features = pose_features.reshape([batch_size*T, 6])
            alpha = torch.cat([lip_features_predict, pose_features], dim=-1)
            directions = self.gen.direction_lipnonlip(alpha)
            rep = torch.LongTensor([T]*batch_size).cuda()
            wa_identity = torch.repeat_interleave(wa_identity, rep, dim=0)
            layer_num = len(feats_identity)
            for i in range(layer_num):
                feats_identity[i] = torch.repeat_interleave(feats_identity[i], rep, dim=0)

            latent = wa_identity + directions

            recon = self.gen.dec(latent, None, feats_identity)# torch.Size([20, 3, 256, 256])

            recon = recon.reshape(batch_size, T, 3,256,256)
            return recon[:,-1]

    def resume(self, resume_ckpt, audio2lip_ckpt):
        print("load model:", resume_ckpt)
        # 单GPU兼容：使用 map_location 避免 device mismatch
        ckpt = torch.load(resume_ckpt, weights_only=False, map_location='cpu')
        ckpt_name = os.path.basename(resume_ckpt)
        # try:
        #     self.start_iter = ckpt["start_iter"] #int(os.path.splitext(ckpt_name)[0])
        # except:
        #     self.start_iter = 0
        try:
            start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            start_iter = 0
        # self.gen.load_state_dict(ckpt["gen"])

        # 兼容新旧两种 checkpoint 格式:
        #   旧格式: 含 'gen'/'g_optim' (完整模型)
        #   新格式: 仅含 'audio2lip' (微调产出, 见 save())
        # 新格式下 gen 保持预训练/初始化状态——这正是设计意图,
        # 因为微调的 dec 权重效果不如预训练, 不应回灌。
        gen_state = ckpt.get('gen')
        if gen_state is not None:
            self.gen.load_state_dict(gen_state)
        else:
            print('[INFO] checkpoint 不含 gen 权重, 跳过加载 (保持预训练/初始化状态)')

        # audio2lip: 优先从训练 checkpoint 加载(微调产出), 回退到独立的 audio2lip_ckpt
        if 'audio2lip' in ckpt:
            self.audio2lip.load_state_dict(ckpt['audio2lip'])
            print('[INFO] audio2lip 已从训练 checkpoint 加载')
        elif audio2lip_ckpt is not None:
            audio_encoder_ckpt = torch.load(audio2lip_ckpt, weights_only=False, map_location='cpu')
            self.audio2lip.load_state_dict(audio_encoder_ckpt['audio2lip'])
            print(f'[INFO] audio2lip 已从 {audio2lip_ckpt} 加载')
        else:
            print('[WARN] 无 audio2lip 权重来源 (checkpoint 与 audio2lip_ckpt 均缺失)')

        if self.dis_weight !=0:
            self.dis.load_state_dict(ckpt["dis"])
        if "g_optim" in ckpt:
            try:
                self.g_optim.load_state_dict(ckpt["g_optim"])
            except:
                print('cannot load pretrained g_optim')
        else:
            print('[INFO] checkpoint 不含 g_optim, 跳过加载 (优化器从零开始)')

        return start_iter

    def save(self, idx, checkpoint_path):
        # 确保所有 CUDA 操作完成，避免序列化时访问违例（Windows 已知问题）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 微调保存策略: 仅保存 audio2lip 权重。
        # 原因: 经测试，微调保存的 Decoder(gen.dec) 权重效果不如预训练权重，
        # 而 gen(含 enc/dec/direction_lipnonlip) 体积巨大，保存它既浪费空间又无实际收益。
        # audio2lip 才是微调的有效产出，推理时配合预训练 gen 使用即可。
        # g_optim 与 gen 参数强绑定，既然不保存 gen，也不再保存 g_optim。
        
        # 关键修复: 将 state_dict 移到 CPU 后再保存，彻底避免 GPU 张量序列化问题
        # Windows WDDM 模式下 torch.save 直接序列化 GPU 张量会导致 access violation
        audio2lip_cpu = {k: v.cpu().clone() for k, v in self.audio2lip.state_dict().items()}
        ckpt = {
            "audio2lip": audio2lip_cpu,
            "args": self.args,
            "start_iter": idx,
        }
        torch.save(ckpt, f"{checkpoint_path}/{str(idx).zfill(6)}.pt")