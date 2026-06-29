import os
import sys

# 解决 Windows OpenMP 冲突（libomp.dll / libiomp5md.dll）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
        self.device = device  # 保存 device 供后续使用（如 get_sync_loss）
        self.batch_size = args.batch_size
        self.dis_weight = args.dis_weight
        self.audio2lip = Audio2Lip().to(device)
        self.train_generator = args.train_generator

        requires_grad(self.audio2lip.audio_encoder, False)

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_lip, args.latent_dim_pose, args.channel_multiplier).to(
            device)
        if self.train_generator:
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

        if args.distributed:
            self.audio2lip = self.audio2lip.module
            if self.train_generator:
                self.gen = self.gen.module
                # self.dis = self.dis.module

        # 参数分组学习率：audio2lip 使用更低 lr，避免破坏预训练权重
        # audio_encoder 已冻结，这里只包含 audio2lip 的可训练参数 (mapping 层)
        audio2lip_params = list(filter(lambda p: p.requires_grad, self.audio2lip.parameters()))

        # 为不同模块设置不同学习率倍数
        # audio2lip: 0.1x (mapping 层学习率保守，避免 latent direction 漂移过大)
        # gen (generator): 1.0x (正常训练)
        # NOTE: 原 StyleGAN2 配置 betas=(0, 0.99) 配合 path length regularization 使用。
        # 本训练无 path length reg，audio2lip 微调是回归任务(L1/VGG/MSE)，
        # beta1=0 无动量导致收敛慢且抖动。改为 (0.5, 0.99)，lr 不再乘 g_reg_ratio。
        betas = (0.5, 0.99)
        param_groups = [
            {
                'params': audio2lip_params,
                'lr': args.lr * 0.1,
                'name': 'audio2lip'
            }
        ]

        if self.train_generator:
            gen_params = list(filter(lambda p: p.requires_grad, self.gen.parameters()))
            param_groups.append({
                'params': gen_params,
                'lr': args.lr,
                'name': 'generator'
            })

        self.g_optim = optim.Adam(
            param_groups,
            betas=betas
        )



        # self.d_optim = optim.Adam(
        #     self.dis.parameters(),
        #     lr=args.lr * d_reg_ratio,
        #     betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        # )

        # self.criterion_vgg = VGGLoss().to('cuda')

        self.criterion_vgg = VGGLoss(device='cuda')

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
                        
                except Exception as e:
                    print(f'[WARN] Failed to load SyncNet from {syncnet_ckpt}: {e}')
            else:
                print(f'[WARN] SyncNet checkpoint not found: {syncnet_ckpt}, sync loss disabled')
                self.sync_weight = 0

            if self.sync_weight != 0:
                for p in self.syncnet.parameters():
                    p.requires_grad = False
                if torch.cuda.is_available():
                    self.syncnet = self.syncnet.to(device)
                    self.syncnet.eval()
                self.logloss = nn.BCELoss()

        self.start_iter = 0
    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, audio_features, lip_features, pose_features, identity_img, target_img, bbox = None, accumulation_steps=1, step_optimizer=True): # torch.Size([64, 5, 80, 16])
        
        self.audio2lip.train()
        self.gen.train()
        
        # 只在梯度累积第一步时清零梯度
        if step_optimizer:
            self.gen.zero_grad()
            self.audio2lip.zero_grad()
        
        G_losses = {}
        # requires_grad(self.audio2lip, True)
        # requires_grad(self.gen.enc, False)
        # requires_grad(self.gen.dec, False)
        # requires_grad(self.gen.fc, False)
        # requires_grad(self.audio2lip.audio_encoder, False)
        # img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = bi['img_a_identity'],bi['img_b_identity'],bi['img_a'],bi['img_b'],bi['imagea_b'],bi['imageb_a']
        # img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = img_a_identity.cuda(), img_b_identity.cuda(), img_a.cuda(), img_b.cuda(), imagea_b.cuda(), imageb_a.cuda()
        batch_size, T = audio_features.shape[0], audio_features.shape[1]
        audio_features = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
        lip_features_predict = self.audio2lip(audio_features, batch_size, T) # batch, T, 20
        # .reshape([bbs*bs, 3, 256, 256]) 
        G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
        G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*0.1

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

        G_losses['recon_vgg_loss'] = self.criterion_vgg(recon, target_img).mean()
        if self.dis_weight !=0:
            G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred)

        G_losses['recon_l1_loss'] = F.l1_loss(recon, target_img)
        
        if self.sync_weight != 0:

            preds = bbox.reshape(batch_size*T, 4)
            preds = preds.to('cuda')/256.
            box_to_feat = torch.from_numpy(np.array([i for i in range(batch_size*T)]))
            gt_bbox = crop_bbox_batch(target_img, preds, box_to_feat, 96)
            pre_bbox = crop_bbox_batch(recon, preds, box_to_feat, 96)

            G_losses['img_l1_sync'] = torch.abs(gt_bbox-pre_bbox).mean()
            pre_bbox = pre_bbox.reshape(batch_size, T, 3, 96, 96).permute(0, 2, 1, 3, 4) # torch.Size([20, 1, 80, 16])
            # SyncNet 训练时 mel 输入为 (B, 1, 80, T*16) - T 帧 mel 沿 W 维度拼接
            # (见 train_syncnet.py 的 mel.view(1, 80, -1))
            # 之前误用 [:,0:1] 只取第 1 帧 (B,1,80,16), 导致 SyncNet 输出垃圾,
            # 余弦相似度 ≈ 0, BCE 被 clamp 到 100, sync 恒为 sync_weight*100=500
            mel_for_sync = audio_features.reshape(batch_size, T, 80, 16).permute(0, 2, 1, 3).reshape(batch_size, 1, 80, T*16)
            value = self.get_sync_loss(mel_for_sync, pre_bbox, self.device).mean()
            G_losses['sync'] = self.sync_weight * value
            
        G_losses_values = [val.mean() for val in G_losses.values()]
        g_loss = sum(G_losses_values)

        # 梯度累积时缩放损失
        if accumulation_steps > 1:
            g_loss = g_loss / accumulation_steps
        
        g_loss.backward()
        
        # 只在累积完成时执行优化器步进
        if step_optimizer:
            self.g_optim.step()
        
        recon = recon.reshape(batch_size, T, 3,256,256)
        return G_losses, recon[:,-1], g_loss

    def get_sync_loss(self, mel, g, device):
        """对比损失 (Wav2Lip 标准)。

        SyncNet 输出已 L2 归一化，audio_embedding(a) 与 face_embedding(v) 的
        cosine_similarity 实为点积，值域 [-1, 1]。

        原实现用 BCELoss(cosine, target=1) 是数学错误的：
        - BCE 要求输入 ∈ [0, 1]（概率），而余弦值可为负 → log(负数) = NaN。
        - 退化时余弦 ≈ 0，BCE 被 clamp 到 ~100，sync 恒为 sync_weight*100。
        改为 InfoNCE 对比损失：拉近正样本(音频↔正确帧)、推开负样本(音频↔错位帧)。
        """
        g = g[:, :, :, g.size(3)//2:]  # torch.Size([B, 3, T, 96, 48]) 取下半脸(嘴部)
        g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=1)  # torch.Size([B, 3*T, 48, 96])
        a, v = self.syncnet(mel, g)  # a, v: (B, 512), 已 L2 normalize

        # 正样本：音频与对应帧的点积 (每行 i 对应 batch 内第 i 个样本)
        pos = (a * v).sum(dim=1)  # (B,)，越大约好

        # 负样本：音频与错位帧的点积 (batch 内 shuffle，避免与自身配对)
        v_neg = torch.roll(v, shifts=1, dims=0)
        neg = (a * v_neg).sum(dim=1)  # (B,)，越小越好

        # InfoNCE：最大化 pos - neg，等价于 softplus(neg - pos).mean()
        # 当 batch=1 时退化为 -pos 的单边损失 (无负样本可比)，仍能提供"对齐"梯度
        sync_loss = F.softplus(neg - pos).mean()
        return sync_loss


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
            # .reshape([bbs*bs, 3, 256, 256]) 
            G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
            G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*0.1

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

            G_losses['recon_vgg_loss'] = self.criterion_vgg(recon, target_img).mean()
            if self.dis_weight !=0:
                G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred)

            G_losses['recon_l1_loss'] = F.l1_loss(recon, target_img)
            
            if self.sync_weight != 0:

                preds = bbox.reshape(batch_size*T, 4)
                preds = preds.to('cuda')/256.
                box_to_feat = torch.from_numpy(np.array([i for i in range(batch_size*T)]))
                gt_bbox = crop_bbox_batch(target_img, preds, box_to_feat, 96)
                pre_bbox = crop_bbox_batch(recon, preds, box_to_feat, 96)

                G_losses['img_l1_sync'] = torch.abs(gt_bbox-pre_bbox).mean()
                pre_bbox = pre_bbox.reshape(batch_size, T, 3, 96, 96).permute(0, 2, 1, 3, 4) # torch.Size([20, 1, 80, 16])
                # SyncNet mel 输入需 T 帧沿 W 维度拼接 (B,1,80,T*16)，与 gen_update 一致。
                # 之前误用 [:,0:1] 只取第 1 帧，导致 eval 日志 sync loss 错误，误导调参。
                mel_for_sync = audio_features.reshape(batch_size, T, 80, 16).permute(0, 2, 1, 3).reshape(batch_size, 1, 80, T*16)
                value = self.get_sync_loss(mel_for_sync, pre_bbox, self.device).mean()
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

        checkpoint = ckpt['gen']
        # new_state_dict = OrderedDict()
        # for key, value in checkpoint.items():
        #     if 'enc.fc.' in key:
        #         if 'enc.fc.4' in key:
        #             continue
        #         name = key.split('enc.fc.')[1]
        #         new_state_dict[name] = value

        self.gen.load_state_dict(checkpoint)

        # new_state_dict = OrderedDict()
        # for key, value in checkpoint.items():
        #     if 'enc.net_app.' in key:
        #         name = key.split('enc.')[1]
        #         new_state_dict[name] = value
        # self.gen.enc.load_state_dict(new_state_dict)
        audio_encoder_ckpt = torch.load(audio2lip_ckpt, weights_only=False, map_location='cpu')
        self.audio2lip.load_state_dict(audio_encoder_ckpt['audio2lip'])
        # new_state_dict = OrderedDict()
        # for key, value in checkpoint.items():
        #     if 'dec.' in key:
        #         if 'dec.direc' in key:
        #             continue
        #         name = key.split('dec.')[1]
        #         new_state_dict[name] = value
        # self.gen.dec.load_state_dict(new_state_dict)
        if self.dis_weight !=0:
            self.dis.load_state_dict(ckpt["dis"])
        
        # 修复 g_optim 加载：过滤掉不匹配的键，只加载匹配的参数
        try:
            g_optim_state = ckpt["g_optim"]
            current_state = self.g_optim.state_dict()
            
            # 过滤状态字典：只保留当前优化器存在的键
            filtered_state = {}
            missing_keys = []
            for key in current_state:
                if key in g_optim_state:
                    # 额外检查形状是否匹配
                    saved_val = g_optim_state[key]
                    cur_val = current_state[key]
                    # param_groups 和 state 是嵌套结构，不是 tensor，不能直接比 shape
                    if isinstance(saved_val, torch.Tensor) and isinstance(cur_val, torch.Tensor):
                        if saved_val.shape == cur_val.shape:
                            filtered_state[key] = saved_val
                        else:
                            missing_keys.append(f"{key} (shape mismatch: {saved_val.shape} vs {cur_val.shape})")
                    elif isinstance(saved_val, (dict, list)) and isinstance(cur_val, (dict, list)):
                        # state dict 和 param_groups — 直接使用保存的值
                        filtered_state[key] = g_optim_state[key]
                    else:
                        missing_keys.append(f"{key} (type mismatch: {type(saved_val).__name__} vs {type(cur_val).__name__})")
                else:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"[WARN] g_optim missing/shape-mismatch keys ({len(missing_keys)}): {missing_keys[:5]}...")
            
            current_state.update(filtered_state)
            self.g_optim.load_state_dict(current_state)
            print(f"[INFO] Loaded g_optim: {len(filtered_state)}/{len(current_state)} keys matched")
        except Exception as e:
            print(f"[WARN] Cannot load pretrained g_optim: {e}")
            print("[INFO] Training will continue with fresh optimizer state (may need lower lr)")
        
        # self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path):
        # 兼容两种调用方式：
        #   1) 传目录:        trainer.save(iter, "logs/.../checkpoint")         -> 保存为 logs/.../checkpoint/000123.pt
        #   2) 传完整文件路径: trainer.save(iter, "logs/.../checkpoint/xxx.pt")  -> 直接用该路径
        if os.path.isdir(checkpoint_path) or not checkpoint_path.endswith('.pt'):
            save_path = os.path.join(checkpoint_path, f"{str(idx).zfill(6)}.pt")
        else:
            save_path = checkpoint_path
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        state = {
            "audio2lip": self.audio2lip.state_dict(),
            "args": self.args
        }
        # train_generator=True 时保存 gen 和 g_optim，否则 gen 未训练无需保存
        if self.train_generator:
            state["gen"] = self.gen.state_dict()
            state["g_optim"] = self.g_optim.state_dict()
        # dis_weight!=0 时保存 dis (d_optim 链路当前未启用，暂不保存)
        if self.dis_weight != 0:
            state["dis"] = self.dis.state_dict()
        torch.save(state, save_path)
