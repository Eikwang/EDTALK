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

        requires_grad(self.audio2lip.audio_encoder, False)

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


        net_parameters = filter(lambda p: p.requires_grad, self.audio2lip.parameters())
        train_parameters = list(net_parameters)
        net_parameters2 = filter(lambda p: p.requires_grad, self.gen.parameters())
        train_parameters += list(net_parameters2)
        # 修正: 原 betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio) 导致 beta1=0.0,
        # Adam 无一阶动量退化为类 RMSProp, 训练不稳定。
        # 改为标准 Adam betas (0.9, 0.999), lr 已通过 g_reg_ratio 缩放。
        self.g_optim = optim.Adam(
            train_parameters,
            lr=args.lr * g_reg_ratio,
            betas=(0.9, 0.999)
        )



        # self.d_optim = optim.Adam(
        #     self.dis.parameters(),
        #     lr=args.lr * d_reg_ratio,
        #     betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        # )

        # self.criterion_vgg = VGGLoss().to('cuda')

        self.criterion_vgg = VGGLoss().to('cuda')

        # VGG 稀疏计算：每 vgg_sparse_freq 个 iter 计算一次 VGG，权重乘以 N 补偿
        self.vgg_sparse_freq = getattr(args, 'vgg_sparse_freq', 0)
        self.disable_vgg = getattr(args, 'disable_vgg', False)
        self._train_iter = 0

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
            G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
            G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*0.5

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

            # VGG 感知损失：支持稀疏计算 + 权重补偿
            # vgg_sparse_freq=0: 每 iter 计算 VGG，权重 0.5
            # vgg_sparse_freq=N: 每 N iter 计算一次 VGG，权重 0.5×N（保持平均贡献不变）
            # disable_vgg=True: 完全关闭 VGG loss
            if self.disable_vgg:
                G_losses['recon_vgg_loss'] = torch.tensor(0.0, device=recon.device)
            elif self.vgg_sparse_freq > 0 and self._train_iter % self.vgg_sparse_freq != 0:
                G_losses['recon_vgg_loss'] = torch.tensor(0.0, device=recon.device)
            else:
                vgg_weight = 0.15 * (self.vgg_sparse_freq if self.vgg_sparse_freq > 0 else 1)
                G_losses['recon_vgg_loss'] = self.criterion_vgg(recon, target_img).mean() * vgg_weight
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
            G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
            G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*0.5

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