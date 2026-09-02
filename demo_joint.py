#!/usr/bin/env python3
"""
联合训练推理脚本
使用联合训练后的模型进行推理

用法：
python demo_joint.py --ckpt_path logs/joint_training/joint_v1/.../checkpoint/joint_epoch_0019_iter_050000.pt --source_path test_data/output.mp4 --audio_driving_path test_data/wy5.wav --save_path res/joint_output.mp4
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import av
from fractions import Fraction
import subprocess
import audio

# 尝试从 imageio-ffmpeg 获取 ffmpeg 路径（无需系统安装）
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = 'ffmpeg'  # 回退到系统 PATH

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from networks.generator_lip_pose import Generator
from networks.audio_encoder import Audio2Lip


class PreloadedVideoReader:
    """一次性解码所有视频帧到 GPU"""
    def __init__(self, vid_path, size=256, device='cuda'):
        self.vid_path = vid_path
        self.size = size
        self.device = device
        
        container = av.open(vid_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        self.fps = float(stream.average_rate)
        
        frames_list = []
        for frame in container.decode(stream):
            img = frame.to_ndarray(format='rgb24')
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = F.interpolate(img.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
            img = (img / 255.0 - 0.5) * 2.0
            frames_list.append(img.squeeze(0))
        
        container.close()
        self.all_frames = torch.stack(frames_list).to(device)
        self.total_frames = self.all_frames.shape[0]
        print(f'==> video info: fps={self.fps:.2f}, total_frames={self.total_frames}, '
              f'GPU memory: {self.all_frames.nelement() * 4 / 1024**2:.1f} MB')
    
    def get_frame(self, idx):
        actual_idx = idx % self.total_frames
        return self.all_frames[actual_idx:actual_idx+1]
    
    def close(self):
        self.all_frames = None


class StreamingVideoWriter:
    """流式视频写入器"""
    def __init__(self, save_path, fps, width=256, height=256):
        self.save_path = save_path
        self.container = av.open(save_path, mode='w')
        self.stream = self.container.add_stream('libx264', rate=Fraction(int(fps), 1))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'
        self.frame_count = 0
    
    def write_frame(self, frame_tensor):
        img = frame_tensor.squeeze(0).clamp(-1, 1).cpu()
        img = (img + 1.0) * 127.5
        img = img.permute(1, 2, 0).to(torch.uint8).numpy()
        av_frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
        self.frame_count += 1
    
    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def get_mel(audio_path):
    """提取mel频谱图
    
    Returns:
        audiox: (bs*T, 1, 80, 16) 形状的音频特征
        bs: batch size (1)
        T: 时间帧数
    """
    wav = audio.load_wav(audio_path, 16000)
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()
    indiv_mels = []
    fps = 25
    syncnet_mel_step_size = 16
    mel_fps = 16000.0 / audio.get_hop_size()
    
    for i in range(num_frames):
        start_frame_num = i - 2
        start_idx = int(mel_fps * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), orig_mel.shape[0]-1) for item in seq]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    
    indiv_mels = np.asarray(indiv_mels)  # T 80 16
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0).cuda()
    source_audio_feature = indiv_mels.type(torch.FloatTensor).cuda()
    
    mel_input = source_audio_feature  # bs T 1 80 16
    bs = mel_input.shape[0]
    T = mel_input.shape[1]
    audiox = mel_input.view(-1, 1, 80, 16)  # bs*T 1 80 16
    
    return audiox, bs, T


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


def conv_feat(features, k_size, sigma=1.0):
    """高斯平滑特征"""
    c = features.shape[1]
    pad = k_size // 2
    k = np.zeros(k_size, dtype=np.float64)
    for x in range(-pad, k_size-pad):
        k[x+pad] = np.exp(-x**2 / (2 * (sigma ** 2)))
    k = k / k.sum()
    k = torch.from_numpy(k).to(features.device).float().unsqueeze(0).unsqueeze(0)
    k = k.repeat(c, 1, 1)
    features = features.unsqueeze(0).permute(0, 2, 1)
    features = F.conv1d(features, k, padding=pad, groups=c)
    features = features.permute(0, 2, 1).squeeze(0)
    return features


class JointDemo(nn.Module):
    """联合训练推理类"""
    def __init__(self, args):
        super(JointDemo, self).__init__()
        self.args = args
        
        print('==> loading joint trained model')
        # 加载联合训练的检查点
        ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
        
        # 加载Audio2Lip
        self.audio2lip = Audio2Lip().cuda()
        self.audio2lip.load_state_dict(ckpt['audio2lip'])
        self.audio2lip.eval()
        
        # 加载Generator
        self.gen = Generator().cuda()
        self.gen.load_state_dict(ckpt['generator'])
        self.gen.eval()
        
        print('==> loading data')
        # 处理视频
        if args.need_crop_pose_video:
            print('==> croping pose_video')
            crop_video_path = os.path.join(os.path.dirname(args.source_path), 'crop_' + os.path.basename(args.source_path))
            crop_cmd = f"python data_preprocess/crop_video.py --inp {args.source_path} --outp {crop_video_path}"
            os.system(crop_cmd)
            args.source_path = crop_video_path
        
        if args.audio_driving_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Warning: The provided audio_driving_path is in video format. Please provide an audio file.")
        
        self.audio, self.bs, self.T = get_mel(args.audio_driving_path)
        self.audio_path = args.audio_driving_path
        self.save_path = args.save_path
        
        self.video_reader = PreloadedVideoReader(args.source_path, size=256, device='cuda')
        self.fps = self.video_reader.fps
        self.total_video_frames = self.video_reader.total_frames
    
    def run(self):
        print('==> running')
        with torch.no_grad():
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            h_start = None
            self.lip_vid_target = self.audio2lip(self.audio, self.bs, self.T)[0]
            self.lip_vid_target = conv_feat(self.lip_vid_target, k_size=3, sigma=1)
            
            total_lip_frames = self.lip_vid_target.size(0)
            total_video_frames = self.total_video_frames
            
            writer = StreamingVideoWriter(self.save_path.replace('.mp4', '_temp.mp4'), fps=self.fps)
            
            for i in tqdm(range(total_lip_frames)):
                img_target_lip = self.lip_vid_target[i:i+1]
                vid_idx = i % total_video_frames if total_video_frames > 0 else 0
                img_target_pose = self.video_reader.get_frame(vid_idx)
                
                img_recon = self.gen.test_from_audio_pose_image(img_target_pose, img_target_lip, img_target_pose, h_start)
                writer.write_frame(img_recon)
            
            writer.close()
            self.video_reader.close()
            
            temp_path = self.save_path.replace('.mp4', '_temp.mp4')
            subprocess.run([
                FFMPEG_PATH, '-y',
                '-i', temp_path,
                '-i', self.audio_path,
                '-vcodec', 'copy',
                self.save_path
            ], check=True)
            os.remove(temp_path)
            
            print(f'==> output saved to {self.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Training Inference')
    parser.add_argument("--ckpt_path", type=str, required=True, help="联合训练检查点路径")
    parser.add_argument("--source_path", type=str, default='test_data/pose_source1.mp4', help="源视频路径")
    parser.add_argument("--audio_driving_path", type=str, default='test_data/teaser.mp3', help="驱动音频路径")
    parser.add_argument("--save_path", type=str, default='res/joint_output.mp4', help="输出视频路径")
    parser.add_argument("--need_crop_pose_video", action='store_true', help="是否裁剪视频")
    
    args = parser.parse_args()
    demo = JointDemo(args)
    demo.run()
