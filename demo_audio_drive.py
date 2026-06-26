"""
优化版: 音频驱动口型 + 序列帧源 + 低内存版
=================================================================
核心特性:
  - 源图像支持「单张图」或「序列帧目录」(支持 "0.png", "1.png" ...)
  - 源图像懒加载: 不一次性载入内存 (处理上万帧也不会 OOM)
  - 输出帧不累积: 逐帧写入 (低内存)
  - 口型: 跟随音频 (Audio2Lip 输出 20 维/帧)

使用方法:
  # 方式1: 单张源图 + 音频 → 姿态固定
  python demo_audio_drive.py ^
      --source_path test_data/identity_source.jpg ^
      --audio_driving_path test_data/mouth_source.wav ^
      --save_path res/output.mp4

  # 方式2: 序列帧目录 + 音频 → 姿态自然变化
  python demo_audio_drive.py ^
      --source_path test_data/frames_dir ^
      --audio_driving_path test_data/mouth_source.wav ^
      --save_path res/output_with_pose.mp4

模型权重:
  - ckpts/EDTalk_lip_pose.pt  (支持口型+姿态的 Generator)
  - ckpts/Audio2Lip.pt        (音频编码器)
"""
import os
import sys
import glob
import shutil
import tempfile
import subprocess
import torch
import torch.nn as nn
from networks.generator_lip_pose import Generator
from networks.audio_encoder import Audio2Lip
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from networks.utils import check_package_installed
from moviepy import VideoFileClip, AudioFileClip
import audio


# =========================================================================
# 流式帧缓冲池 - 滑动窗口预加载 (解决帧错乱问题)
# =========================================================================

class StreamingFrameBuffer:
    """
    流式帧缓冲池 - 固定窗口滑动预加载
    ===========================
    特性:
      - 固定窗口大小 → 显存可控 (~20MB/30帧)
      - 预分配 GPU 内存 → 无运行时分配
      - 滑动更新 → 支持任意长度流式输入
      - 避免每帧从磁盘读取导致的时序问题
    """

    def __init__(self, frames_dir, window_size=30, size=256, device='cuda'):
        self.frames_dir = frames_dir
        self.window_size = window_size
        self.size = size
        self.device = device

        # 扫描所有帧文件
        frame_files = (
            glob.glob(os.path.join(frames_dir, '*.png')) +
            glob.glob(os.path.join(frames_dir, '*.jpg'))
        )

        def sort_key(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        frame_files.sort(key=sort_key)

        self.frame_files = frame_files
        self.total_frames = len(frame_files)

        # 预分配 GPU 连续内存 (整个 buffer)
        # shape: [window_size, 3, H, W], dtype: float32
        self.buffer = torch.empty(
            window_size, 3, size, size,
            dtype=torch.float32, device=device
        )

        # 当前窗口信息
        self._current_window_start = -1

        # 加载第一个窗口
        self._preload_window(0)

        # 显存占用计算
        mem_mb = window_size * 3 * size * size * 4 / (1024 ** 2)
        print(f'    [StreamingFrameBuffer] window_size={window_size}, '
              f'memory≈{mem_mb:.1f}MB, total_frames={self.total_frames}')

    def _preload_window(self, window_start):
        """预加载窗口帧到 GPU buffer"""
        for i in range(self.window_size):
            idx = (window_start + i) % self.total_frames
            frame_path = self.frame_files[idx]

            # CPU 读取并预处理 (单帧，内存很小)
            img = Image.open(frame_path).convert('RGB').resize((self.size, self.size))
            img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)  # CHW
            img = (img / 127.5 - 1.0)  # [-1, 1]

            # 直接写入预分配的 GPU 内存 (无新分配)
            self.buffer[i] = torch.from_numpy(img).to(self.device, non_blocking=True)

        self._current_window_start = window_start

    def get_frame(self, idx):
        """获取第 idx 帧 → [1, 3, H, W]"""
        # 计算需要的窗口
        window_start = (idx // self.window_size) * self.window_size

        # 窗口滑动：需要加载新窗口
        if window_start != self._current_window_start:
            self._preload_window(window_start)

        # 在 buffer 中的位置
        buffer_idx = idx % self.window_size
        return self.buffer[buffer_idx:buffer_idx+1]


# =========================================================================
# 单张图像模式 - 保持原样
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
# 音频加载与 Mel 频谱
# =========================================================================

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
    """音频 → Mel 频谱序列"""
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
    source_audio_feature = indiv_mels.to(dtype=torch.float32)
    mel_input = source_audio_feature
    bs = mel_input.shape[0]
    T = mel_input.shape[1]
    audiox = mel_input.view(-1, 1, 80, 16)
    return audiox, bs, T


def audio_preprocessing(wav_path):
    return get_mel(wav_path)


# =========================================================================
# 口型系数平滑
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
# =========================================================================

def write_frame_to_png(img_recon, frame_idx, temp_dir):
    """
    把推理输出的单帧保存成 temp_dir/{frame_idx:06d}.png
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

        # --- 2. Generator (支持口型+姿态)
        self.gen = Generator().cuda()
        ckpt_gen = torch.load(args.model_path, map_location=lambda storage, loc: storage, weights_only=False)
        self.gen.load_state_dict(ckpt_gen['gen'])
        self.gen.eval()
        print(f'    Generator: OK ({os.path.basename(args.model_path)})')

        # ===== 加载输入 =====
        print('==> [Step 2/4] 扫描输入数据')

        # --- 源图像: 根据类型选择加载方式
        self.source_files, self.is_single_source, self.size = scan_frames(
            args.source_path, args.size)
        self.num_source = len(self.source_files)

        # 序列帧模式: 使用流式缓冲池 (滑动窗口)
        # 单张图像模式: 保持原样 (每帧返回同一张图)
        if not self.is_single_source and hasattr(args, 'window_size'):
            window_size = args.window_size
        else:
            window_size = 1  # 单张图模式

        self.window_size = getattr(args, 'window_size', 30)  # 默认窗口 30 帧

        if not self.is_single_source:
            # 序列帧目录: 使用流式缓冲池
            self.frame_buffer = StreamingFrameBuffer(
                args.source_path,
                window_size=self.window_size,
                size=self.size,
                device='cuda'
            )
            print(f'    [source: streaming buffer] window_size={self.window_size}')
        else:
            # 单张图像: 预加载到 GPU
            self.single_img = img_preprocessing(self.source_files[0], self.size)
            print(f'    [source: single image] {os.path.basename(self.source_files[0])}')

        # --- 音频 (在 __init__ 中不加载大音频数据，保留到 run() 中处理，避免长时间不使用)
        self.audio_path = args.audio_driving_path

        self.save_path = args.save_path

    def _get_source_frame(self, frame_idx):
        """
        根据输出帧索引获取对应的源图像:
          - 单张源图 → 总是用第 0 张 (姿态固定)
          - N 张源图 → 使用流式缓冲池 (从 GPU buffer 取)
        """
        if self.is_single_source:
            return self.single_img

        # 使用流式缓冲池获取帧
        return self.frame_buffer.get_frame(frame_idx)

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

            os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)

            # ===== 步骤2: 逐帧推理 + 写入 PNG =====
            print('==> [Step 4/4] 逐帧推理 (低内存模式)')
            temp_dir = tempfile.mkdtemp(prefix='edtalk_frames_')

            try:
                h_start = None
                for i in tqdm(range(total_frames), desc='    生成+写入帧'):
                    img_target_lip = lip_vid_target[i:i+1]
                    img_source_i = self._get_source_frame(i)

                    img_recon = self.gen.test_from_audio_pose_image(
                        img_source_i,
                        img_target_lip,
                        img_source_i,
                        h_start
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
    parser = argparse.ArgumentParser(description='EDTalk: 音频驱动口型 + 序列帧源图像 (低内存版)')
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

    parser.add_argument('--save_path', type=str, default='res/demo_audio_drive.mp4')
    parser.add_argument('--audio2lip_model_path', type=str, default='ckpts/Audio2Lip.pt')
    parser.add_argument('--model_path', type=str, default='ckpts/EDTalk_lip_pose.pt',
                        help='支持口型+姿态的 Generator 权重')
    parser.add_argument('--window_size', type=int, default=30,
                        help='流式帧缓冲池窗口大小 (默认30帧, 约23MB显存)')

    args = parser.parse_args()

    demo = Demo(args)
    demo.run()



