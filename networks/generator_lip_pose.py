from torch import nn
from .encoder import *
from .styledecoder import Synthesis, Synthesis_lip_pose
import torch

class Direction(nn.Module):
    def __init__(self, lip_dim, pose_dim):
        super(Direction, self).__init__()
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.weight = nn.Parameter(torch.randn(512, lip_dim+pose_dim))

    def _compute_Q(self):
        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)
        if Q.dtype != torch.float32:
            Q = Q.to(torch.float32)
        return Q

    def forward(self, input):
        Q = self._compute_Q()
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            if input_diag.dtype != torch.float32:
                input_diag = input_diag.to(torch.float32)
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out

    def get_shared_out(self, input):
        Q = self._compute_Q()
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            out = torch.matmul(input_diag, Q.T)
            return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:], dim=1)
        return pose_latent


class Pose_Direction(nn.Module):
    def __init__(self, pose_dim):
        super(Pose_Direction, self).__init__()
        self.pose_dim = pose_dim
        self.weight = nn.Parameter(torch.randn(512, pose_dim))

    def _compute_Q(self):
        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)
        if Q.dtype != torch.float32:
            Q = Q.to(torch.float32)
        return Q

    def forward(self, input):
        Q = self._compute_Q()
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            if input_diag.dtype != torch.float32:
                input_diag = input_diag.to(torch.float32)
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out

    def get_shared_out(self, input):
        Q = self._compute_Q()
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            out = torch.matmul(input_diag, Q.T)
            return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:], dim=1)
        return pose_latent
    

class Lip_Direction(nn.Module):
    def __init__(self, lip_dim):
        super(Lip_Direction, self).__init__()
        self.lip_dim = lip_dim
        self.weight = nn.Parameter(torch.randn(512, lip_dim))

    def _compute_Q(self, pose_weight=None):
        if pose_weight is not None:
            weight = torch.cat([self.weight, pose_weight], -1)
        else:
            weight = self.weight
        weight = weight + 1e-8
        Q, R = torch.linalg.qr(weight)
        if Q.dtype != torch.float32:
            Q = Q.to(torch.float32)
        return Q

    def forward(self, input, pose_weight):
        Q = self._compute_Q(pose_weight)
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out

    def get_shared_out(self, input):
        Q = self._compute_Q()
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            out = torch.matmul(input_diag, Q.T)
            return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:], dim=1)
        return pose_latent

class Generator(nn.Module):
    def __init__(self, size=256, style_dim=512, lip_dim=20, pose_dim=6, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis_lip_pose(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        # self.direction = Direction(motion_dim)
        self.direction_lipnonlip = Direction(lip_dim, pose_dim)
        # motion network
        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(3):
            fc.append(EqualLinear(style_dim, style_dim))
        self.fc = nn.Sequential(*fc)
        # self.source_fc = EqualLinear(style_dim, motion_dim)

        lip_fc = [EqualLinear(style_dim, style_dim)]
        lip_fc.append(EqualLinear(style_dim, style_dim))
        lip_fc.append(EqualLinear(style_dim, lip_dim))
        self.lip_fc = nn.Sequential(*lip_fc)

        pose_fc = [EqualLinear(style_dim, style_dim)]
        pose_fc.append(EqualLinear(style_dim, style_dim))
        pose_fc.append(EqualLinear(style_dim, pose_dim))
        self.pose_fc = nn.Sequential(*pose_fc)


    def forward(self, img_source, img_drive, h_start=None):
        wa, wa_t, feats, feats_t = self.enc(img_source, img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)
        alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def orthogonality_loss(self):
        """计算 Direction 权重矩阵的正交正则化损失

        目标: W^T @ W 接近单位矩阵，确保 lip 和 pose 方向正交
        """
        W = self.direction_lipnonlip.weight  # [512, 26]
        WtW = W.T @ W  # [26, 26]
        I = torch.eye(WtW.shape[0], device=WtW.device, dtype=WtW.dtype)
        return F.mse_loss(WtW, I)

    def test_from_audio_pose_image(self, img_source, alpha_D_lip, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_pose = self.pose_fc(shared_fc)


        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_EDTalk_V(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, wa_t, feats, _ = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_lip,_, _,_ = self.enc(lip_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_pose = self.pose_fc(shared_fc)
        shared_fc = self.fc(wa_t_lip)
        alpha_D_lip = self.lip_fc(shared_fc)


        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def get_lip_pose_feature(self, img_source):
        """
        从单张图像提取 lip 和 pose 特征（用于训练数据预处理）
        
        Args:
            img_source: 预处理后的人脸图像 [B, 3, 256, 256]
            
        Returns:
            lip_feat: [B, 20] 口型特征
            pose_feat: [B, 6] 姿态特征
        """
        # Encoder 需要两个参数: (input_source, input_target)
        # input_target 设为 None 表示不需要驱动图像
        # 返回: h_source, h_target(None), feats, feats_t(None)
        wa_t_p, _, _, _ = self.enc(img_source, None)
        
        # 通过 FC 层提取特征
        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)
        alpha_D_lip = self.lip_fc(shared_fc_p)
        
        return alpha_D_lip, alpha_D_pose