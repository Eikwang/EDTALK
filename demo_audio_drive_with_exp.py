"""
优化版 v2: 音频驱动口型 + 预定义表情 + 序列帧源 + 低内存版
=================================================================
核心特性:
  - 源图像支持「单张图」或「序列帧目录」(支持 "0.png", "1.png" ...)
  - 源图像懒加载: 不一次性载入内存 (处理上万帧也不会 OOM)
  - 输出帧不累积: 生成一张写入一张 (ffmpeg pipe)
  - 不使用 torchvision.io.write_video (该函数在新版 av 库上崩溃)
  - 口型: 跟随音频 (Audio2Lip 输出 20 维/帧)
  - 表情: 固定为指定情绪 (ckpts/predefined_exp_weights/*.npy)

使用方法:
  # 方式1: 单张源图 + 音频 → 姿态固定
  python demo_audio_drive_with_exp.py ^
      --source_path test_data/identity_source.jpg ^
      --audio_driving_path test_data/mouth_source.wav ^
      --exp_type happy ^
      --save_path res/output.mp4

  # 方式2: 序列帧目录 + 音频 → 姿态自然变化
  python demo_audio_drive_with_exp.py ^
      --source_path test_data/frames_dir ^
      --audio_driving_path test_data/mouth_source.wav ^
      --exp_type happy ^
      --save_path res/output_with_pose.mp4

模型权重:
  - ckpts/EDTalk.pt          (完整版 Generator, 支持表情)
  - ckpts/Audio2Lip.pt       (音频编码器)
  - ckpts/predefined_exp_weights/{exp_type}.npy
"""
import os
import sys
import glob
import shutil
import tempfile
import subprocess
import torch
import torch.nn as nn
from networks.generator import Generator
from networks.audio_encoder import Audio2Lip
import argparse
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F


# =========================================================================
# 图像加载与预处理
# 注意: 源图像懒加载 — 只保存文件路径列表, 推理时按需读取单张
# =========================================================================

def load_image(filename, size):
    """加载单张图像并调整尺寸: PIL → numpy (C,H,W), 范围 [0,1]"""
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    return img / 255.0


def img_preprocessing(img_path, size):
    """单张图像预处理: 加载 → torch tensor → 归一化到 [-1, 1]"""
    img = load_image(img_path, size)
    img = torch.from_numpy(img).unsqueeze(0).float()   # [1, C, H, W]
    imgs_norm = (img - 0.5) * 2.0
    return imgs_norm.cuda()


def scan_frames(frames_path_or_dir, size):
    """
    扫描源图像路径 (单张图像 或 目录)

    返回:
        source_files: 源图像文件路径列表 (懒加载用)
        is_single: True 表示是单张图
        调用者通过 load_source_frame(i) 按需读取
    """
    if os.path.isdir(frames_path_or_dir):
        frame_files = (
            glob.glob(os.path.join(frames_path_or_dir, '*.png')) +
            glob.glob(os.path.join(frames_path_or_dir, '*.jpg'))
        )

        def _frame_sort_key(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        frame_files.sort(key=_frame_sort_key)

        if len(frame_files) == 0:
            raise FileNotFoundError(f"目录 {frames_path_or_dir} 中未找到 .png 或 .jpg 文件")

        print(f'    [source: directory] 找到 {len(frame_files)} 帧, 从 {os.path.basename(frame_files[0])} 开始')
        return frame_files, False, size

    else:
        if not os.path.isfile(frames_path_or_dir):
            raise FileNotFoundError(f"源图像/目录不存在: {frames_path_or_dir}")
        print(f'    [source: single image] {os.path.basename(frames_path_or_dir)}')
        return [frames_path_or_dir], True, size


# =========================================================================
# 音频加载与 Mel 频谱 (与 EDTalk 原始保持一致)
# =========================================================================

import audio


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps
    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)
    return audio_length, num_frames


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav


def get_mel(audio_path):
    """
    音频 → Mel 频谱序列
    """
    wav = audio.load_wav(audio_path, 16000)
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)

    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()
    indiv_mels = []
    fps = 25
    syncnet_mel_step_size = 16

    for i in range(num_frames):
        start_frame_num = i - 2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
        m = spec[seq, :]
        indiv_mels.append(m.T)

    indiv_mels = np.asarray(indiv_mels)
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0).cuda()
    source_audio_feature = indiv_mels.type(torch.FloatTensor).cuda()
    mel_input = source_audio_feature
    bs = mel_input.shape[0]
    T = mel_input.shape[1]
    audiox = mel_input.view(-1, 1, 80, 16)
    return audiox, bs, T


def audio_preprocessing(wav_path):
    return get_mel(wav_path)


# =========================================================================
# 口型系数平滑 (与 EDTalk 原始保持一致)
# =========================================================================

def conv_feat(features, k_size, weight=None, sigma=1.0):
    c = features.shape[1]
    if weight is None:
        pad = k_size // 2
        k = np.zeros(k_size).astype(np.float64)
        for x in range(-pad, k_size - pad):
            k[x + pad] = np.exp(-x**2 / (2 * (sigma ** 2)))
        k = k / k.sum()
    else:
        k_size = len(weight)
        k = np.array(weight)
        pad = k_size // 2
    k = torch.from_numpy(k).to(features.device).float().unsqueeze(0).unsqueeze(0)
    k = k.repeat(c, 1, 1)
    features = features.unsqueeze(0).permute(0, 2, 1)
    features = F.conv1d(features, k, padding=pad, groups=c)
    features = features.permute(0, 2, 1).squeeze(0)
    return features


# =========================================================================
# 视频写入: 直接写 PNG 序列到临时目录 → ffmpeg 合成 → 与音频合并
# 优势: (1) 不累积输出帧在内存, (2) 避开 torchvision.io.write_video 崩溃
# =========================================================================

def write_frame_to_png(img_recon, frame_idx, temp_dir):
    """
    把推理输出的单帧(Generator 输出 ([1, 3, H, W], [-1, 1]) 保存成 temp_dir/{frame_idx:06d}.png
    """
    img = img_recon.clamp(-1, 1).cpu()
    img = (img + 1.0) * 127.5  # [-1,1] → [0,255]
    img = img[0].permute(1, 2, 0).numpy().astype(np.uint8)
    pil_img = Image.fromarray(img)
    out_path = os.path.join(temp_dir, f'{frame_idx:06d}.png')
    pil_img.save(out_path)
    return out_path


def compile_video_from_pngs(temp_dir, total_frames, fps, save_path):
    """用 ffmpeg 把临时目录下的 PNG 序列合成为 mp4 (不含音频)"""
    pattern = os.path.join(temp_dir, '%06d.png')
    temp_mp4 = save_path.replace('.mp4', '_no_audio.mp4')

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        temp_mp4
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_tail = result.stderr[-500:] if result.stderr else ''
        print(f'    [ffmpeg 视频合成 stderr]: {err_tail}')
        raise RuntimeError('ffmpeg 视频合成失败，请检查 ffmpeg 是否安装并在 PATH 中')
    return temp_mp4


def merge_audio(video_no_audio, audio_path, save_path):
    """把音频 mux 到视频中"""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_no_audio,
        '-i', audio_path,
        '-vcodec', 'copy',
        '-acodec', 'aac',
        save_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_tail = result.stderr[-500:] if result.stderr else ''
        print(f'    [ffmpeg 音视频合并 stderr]: {err_tail}')
        raise RuntimeError('ffmpeg 音视频合并失败')


# =========================================================================
# 主推理类
# =========================================================================

class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()
        self.args = args
        self.fps = 25

        print('==> [Step 1/4] 加载模型')

        # --- 1. Audio2Lip (音频 → 口型系数)
        self.audio2lip = Audio2Lip().cuda()
        ckpt_a2l = torch.load(args.audio2lip_model_path, map_location=lambda storage, loc: storage, weights_only=False)
        self.audio2lip.load_state_dict(ckpt_a2l['audio2lip'])
        self.audio2lip.eval()
        print(f'    Audio2Lip: OK ({os.path.basename(args.audio2lip_model_path)})')

        # --- 2. 完整版 Generator (支持 36 维 alpha_D: 20+6+10)
        self.gen = Generator(
            args.size, args.latent_dim_style, args.latent_dim_lip,
            args.latent_dim_pose, args.latent_dim_exp, args.channel_multiplier
        ).cuda()
        ckpt_gen = torch.load(args.model_path, map_location=lambda storage, loc: storage, weights_only=False)
        self.gen.load_state_dict(ckpt_gen['gen'])
        self.gen.eval()
        print(f'    Generator: OK ({os.path.basename(args.model_path)})')

        # ===== 加载输入 =====
        print('==> [Step 2/4] 扫描输入数据')

        # --- 源图像: 懒加载 (只保存文件路径, 推理时按需读取)
        self.source_files, self.is_single_source, self.size = scan_frames(
            args.source_path, args.size)
        self.num_source = len(self.source_files)

        # --- 音频 (在 __init__ 中不加载大音频数据，保留到 run() 中处理，避免长时间不使用)
        self.audio_path = args.audio_driving_path

        # --- 表情: 预定义 .npy
        exp_path = os.path.join('ckpts/predefined_exp_weights', args.exp_type + '.npy')
        if not os.path.isfile(exp_path):
            raise FileNotFoundError(f'表情权重文件不存在: {exp_path}')
        self.alpha_D_exp = np.load(exp_path)
        self.alpha_D_exp = torch.from_numpy(self.alpha_D_exp).cuda()

        if args.exp_strength != 1.0:
            self.alpha_D_exp = self.alpha_D_exp * args.exp_strength
            print(f'    [expression] {args.exp_type} (强度 × {args.exp_strength})')
        else:
            print(f'    [expression] {args.exp_type}')

        self.save_path = args.save_path

    def _get_source_frame(self, frame_idx):
        """
        根据输出帧索引获取对应的源图像:
          - 单张源图 → 总是用第 0 张 (姿态固定)
          - N 张源图 → 循环播放 (frame_idx % num_source)
        """
        if self.is_single_source:
            return img_preprocessing(self.source_files[0], self.size)

        # 循环播放源图像，避免超出部分姿态冻结
        idx = frame_idx % self.num_source
        return img_preprocessing(self.source_files[idx], self.size)

    def run(self):
        print('==> [Step 3/4] 处理音频')
        with torch.no_grad():
            # 处理音频 (放在 run() 中，避免初始化时的长时间占用)
            audio, bs, T_audio = audio_preprocessing(self.audio_path)
            print(f'    [audio] {os.path.basename(self.audio_path)} → {T_audio} 帧')

            # ===== 步骤1: 音频 → 口型系数序列 =====
            lip_vid_target = self.audio2lip(audio, bs, T_audio)[0]
            lip_vid_target = conv_feat(lip_vid_target, k_size=3, sigma=1)

            total_frames = lip_vid_target.size(0)
            print(f'    总输出帧数: {total_frames} (源图像: {self.num_source} 张)')

            # 检查帧数匹配情况
            if self.num_source > 1:
                if total_frames > self.num_source:
                    print(f'    ⚠️  音频帧数 > 源图像帧数, 将循环播放源图像 (循环 {total_frames // self.num_source} 次 + {total_frames % self.num_source} 帧)')
                elif total_frames < self.num_source:
                    print(f'    ℹ️  只使用前 {total_frames} 张源图像')
                else:
                    print(f'    ✅ 帧数匹配')

            # ===== 步骤1.5: 姿态处理（禁用姿态变化）=====
            print('    [pose] 禁用姿态变化（姿态系数设为零向量）')
            # 姿态系数为零向量，完全不进行姿态变换
            pose_dim = 6  # 姿态维度
            pose_coeffs_smooth = torch.zeros(total_frames, pose_dim).cuda()

            os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)

            # ===== 步骤2: 逐帧推理 + 写入 PNG =====
            print('==> [Step 4/4] 逐帧推理 (低内存模式)')
            temp_dir = tempfile.mkdtemp(prefix='edtalk_frames_')

            try:
                for i in tqdm(range(total_frames), desc='    生成+写入帧'):
                    img_target_lip = lip_vid_target[i:i+1]
                    img_source_i = self._get_source_frame(i)
                    alpha_D_pose_i = pose_coeffs_smooth[i:i+1]

                    # 使用平滑后的姿态系数
                    img_recon = self.gen.test_EDTalk_A_use_exp_weight_with_pose(
                        img_source_i,
                        img_target_lip,
                        alpha_D_pose_i,
                        self.alpha_D_exp,
                        None
                    )

                    write_frame_to_png(img_recon, i, temp_dir)

                # ===== 步骤3: 合成视频 + 合并音频 =====
                print(f'    [ffmpeg] 正在用 PNG 序列合成视频...')
                temp_mp4 = compile_video_from_pngs(temp_dir, total_frames, self.fps, self.save_path)

                print(f'    [ffmpeg] 合并音频...')
                merge_audio(temp_mp4, self.audio_path, self.save_path)

                # 清理临时文件
                if os.path.isfile(temp_mp4):
                    os.remove(temp_mp4)
            finally:
                # 无论成功失败，都清理临时 PNG 目录
                shutil.rmtree(temp_dir, ignore_errors=True)

            print(f'==> 完成! 视频已保存: {self.save_path}')


# =========================================================================
# 入口
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDTalk: 音频驱动口型 + 预定义表情 + 序列帧源图像 (低内存版)')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--latent_dim_style', type=int, default=512)
    parser.add_argument('--latent_dim_lip', type=int, default=20)
    parser.add_argument('--latent_dim_pose', type=int, default=6)
    parser.add_argument('--latent_dim_exp', type=int, default=10)

    parser.add_argument('--source_path', type=str, default='test_data/identity_source.jpg',
                        help='源图像路径 (单张 .png/.jpg) 或 目录 (自动读取其中所有 .png/.jpg 并按文件名排序)')
    parser.add_argument('--audio_driving_path', type=str, default='test_data/mouth_source.wav',
                        help='驱动音频文件 (.wav)')

    parser.add_argument('--exp_type', type=str, default='happy',
                        choices=['angry', 'contempt', 'disgusted', 'fear',
                                 'happy', 'sad', 'surprised'],
                        help='表情类型, 从 ckpts/predefined_exp_weights/{exp_type}.npy 加载')
    parser.add_argument('--exp_strength', type=float, default=1.0,
                        help='表情强度 (0.5=较温和, 1.0=默认, 1.5=较夸张)')

    parser.add_argument('--save_path', type=str, default='res/demo_audio_drive_with_exp.mp4')
    parser.add_argument('--audio2lip_model_path', type=str, default='ckpts/Audio2Lip.pt')
    parser.add_argument('--model_path', type=str, default='ckpts/EDTalk.pt',
                        help='必须是完整版 Generator 权重 (支持表情)')

    args = parser.parse_args()

    demo = Demo(args)
    demo.run()
