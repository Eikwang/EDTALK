import os, sys
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
import av
from fractions import Fraction

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


class PreloadedVideoReader:
    """启动时一次性解码所有视频帧到 GPU 显存，推理时直接索引，零延迟"""

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
            img = frame.to_ndarray(format='rgb24')  # H x W x 3, uint8
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # 3 x H x W
            img = F.interpolate(img.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
            img = (img / 255.0 - 0.5) * 2.0  # [-1, 1]
            frames_list.append(img.squeeze(0))

        container.close()

        # [N, 3, H, W] 一次性存入 GPU
        self.all_frames = torch.stack(frames_list).to(device)
        self.total_frames = self.all_frames.shape[0]
        print(f'==> video info: fps={self.fps:.2f}, total_frames={self.total_frames}, '
              f'GPU memory: {self.all_frames.nelement() * 4 / 1024**2:.1f} MB')

    def get_frame(self, idx):
        """获取第 idx 帧（支持循环：idx 超出总帧数时取模），返回 [1, 3, H, W]"""
        actual_idx = idx % self.total_frames
        return self.all_frames[actual_idx:actual_idx+1]  # [1, 3, H, W]

    def close(self):
        self.all_frames = None


class StreamingVideoWriter:
    """基于 PyAV 的流式视频写入器，逐帧写入，避免内存中积攒所有帧"""

    def __init__(self, save_path, fps, width=256, height=256):
        self.save_path = save_path
        self.container = av.open(save_path, mode='w')
        self.stream = self.container.add_stream('libx264', rate=Fraction(int(fps), 1))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'
        self.frame_count = 0

    def write_frame(self, frame_tensor):
        """写入一帧 [1, 3, H, W] 的 tensor（[-1,1] 范围）"""
        img = frame_tensor.squeeze(0).clamp(-1, 1).cpu()
        img = (img + 1.0) * 127.5  # [-1,1] → [0,255]
        img = img.permute(1, 2, 0).to(torch.uint8).numpy()  # H x W x 3, uint8
        av_frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self):
        # Flush
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()

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

    wav = audio.load_wav(audio_path, 16000) 
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()         # nframes 80
    indiv_mels = []
    fps = 25
    syncnet_mel_step_size = 16


    for i in range(num_frames):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)         # T 80 16
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0).cuda()
    source_audio_feature = indiv_mels.type(torch.FloatTensor).cuda()

    mel_input = source_audio_feature                       # bs T 1 80 16
    bs = mel_input.shape[0]
    T = mel_input.shape[1]
    audiox = mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16

    return audiox, bs, T


def audio_preprocessing(wav_path):
    source_audio_feature, bs, T = get_mel(wav_path)

    return source_audio_feature, bs, T

class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args
        model_path = args.model_path
        audio2lip_model_path = args.audio2lip_model_path
        print('==> loading model')
        self.audio2lip = Audio2Lip().cuda()
        weight = torch.load(audio2lip_model_path, map_location=lambda storage, loc: storage, weights_only=False)['audio2lip']
        self.audio2lip.load_state_dict(weight)
        self.audio2lip.eval()
        self.gen = Generator().cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()
        print('==> loading data')


        if args.need_crop_pose_video:
            print('==> croping pose_video')
            crop_video_path = os.path.join(os.path.dirname(args.source_path), 'crop_'+os.path.basename(args.source_path))
            crop_cmd = f"python data_preprocess/crop_video.py --inp {args.source_path} --outp {crop_video_path}"
            os.system(crop_cmd)

            args.source_path = crop_video_path
        

        if args.audio_driving_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Warning: The provided audio_driving_path is in video format. Please provide an audio file.")

        self.audio, self.bs, self.T = audio_preprocessing(args.audio_driving_path)
        self.audio_path = args.audio_driving_path
        self.save_path = args.save_path

        self.video_reader = PreloadedVideoReader(args.source_path, size=256, device='cuda')
        self.fps = self.video_reader.fps
        self.total_video_frames = self.video_reader.total_frames

    def run(self):

        print('==> running')
        with torch.no_grad():
            # self.save_path = args.save_path
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            h_start = None
            self.lip_vid_target = self.audio2lip(self.audio, self.bs, self.T)[0]
            self.lip_vid_target = conv_feat(self.lip_vid_target, k_size=3, sigma=1) # torch.Size([372, 500])

            total_lip_frames = self.lip_vid_target.size(0)
            total_video_frames = self.total_video_frames

            # 流式写入器，逐帧写入，不占用内存
            writer = StreamingVideoWriter(self.save_path.replace('.mp4','_temp.mp4'), fps=self.fps)

            # 时序一致性平滑状态 (方案D)
            prev_frame = None
            use_temporal_smooth = getattr(self.args, 'temporal_smooth', False)
            smooth_weight = getattr(self.args, 'smooth_weight', 0.7)
            # 预定义嘴部区域 mask: 图像底部 40% 作为嘴部区域
            mouth_ratio = 0.4

            for i in tqdm(range(total_lip_frames)):
                img_target_lip = self.lip_vid_target[i:i+1]

                # 流式读取源视频帧，自动循环
                vid_idx = i % total_video_frames if total_video_frames > 0 else 0
                img_target_pose = self.video_reader.get_frame(vid_idx)

                img_recon = self.gen.test_from_audio_pose_image(img_target_pose, img_target_lip, img_target_pose, h_start)

                # 时序一致性平滑 (方案D)
                if use_temporal_smooth and prev_frame is not None:
                    # 创建嘴部区域 mask: 顶部是背景, 底部 mouth_ratio 是嘴部
                    H = img_recon.shape[2]
                    mouth_start = int(H * (1.0 - mouth_ratio))
                    mouth_mask = torch.zeros(1, 1, H, img_recon.shape[3], device=img_recon.device)
                    mouth_mask[:, :, mouth_start:, :] = 1.0
                    # 背景区域: 用历史帧平滑; 嘴部区域: 保留当前帧
                    img_recon = (1.0 - mouth_mask) * (smooth_weight * prev_frame + (1.0 - smooth_weight) * img_recon) \
                                + mouth_mask * img_recon

                prev_frame = img_recon.detach().clone()

                writer.write_frame(img_recon)

            writer.close()
            self.video_reader.close()
            
            temp_path = self.save_path.replace('.mp4','_temp.mp4')
            cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (temp_path, self.audio_path, self.save_path)
            os.system(cmd)
            os.remove(temp_path)

            if self.args.face_sr and check_package_installed('gfpgan'):
                from face_sr.face_enhancer import enhancer_list
                import imageio

                temp_512_path = self.save_path.replace('.mp4','_512.mp4')

                # Super-resolution
                imageio.mimsave(temp_512_path + '.tmp.mp4', enhancer_list(self.save_path, method='gfpgan', bg_upsampler=None), fps=float(25), codec='libx264')
                
                # Merge audio and video
                video_clip = VideoFileClip(temp_512_path + '.tmp.mp4')
                audio_clip = AudioFileClip(self.save_path)
                final_clip = video_clip.with_audio(audio_clip)
                final_clip.write_videofile(temp_512_path, codec='libx264', audio_codec='aac')
                
                os.remove(temp_512_path + '.tmp.mp4')


def conv_feat(features, k_size, weight=None, sigma=1.0):
    c = features.shape[1] # torch.Size([101, 500])
    if weight is None:
        pad = k_size // 2
        k = np.zeros(k_size, dtype=np.float64)
        for x in range(-pad, k_size-pad):
            k[x+pad] = np.exp(-x**2 / (2 * (sigma ** 2)))
        k = k / k.sum()
        print(k) # [0.27406862 0.45186276 0.27406862]
    else:
        k_size = len(weight)
        k = np.array(weight)
        pad = k_size // 2
        print(k)
    
    k = torch.from_numpy(k).to(features.device).float().unsqueeze(0).unsqueeze(0)
    k = k.repeat(c, 1, 1)
    features = features.unsqueeze(0).permute(0, 2, 1) # [1, 512, n]
    features = F.conv1d(features, k, padding=pad, groups=c)
    features = features.permute(0, 2, 1).squeeze(0)

    return features

if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_lip", type=int, default=20)
    parser.add_argument("--latent_dim_pose", type=int, default=6)
    parser.add_argument("--latent_dim_exp", type=int, default=10)
    parser.add_argument("--source_path", type=str, default='test_data/pose_source1.mp4')
    parser.add_argument("--audio_driving_path", type=str, default='test_data/teaser.mp3')
    parser.add_argument("--save_path", type=str, default='res/demo_change_a_video_lip_teaser.mp4')
    parser.add_argument("--audio2lip_model_path", type=str, default='ckpts/Audio2Lip.pt')
    parser.add_argument("--model_path", type=str, default='ckpts/EDTalk_lip_pose.pt')
    parser.add_argument('--face_sr', action='store_true', help='Face super-resolution (Optional). Please install GFPGAN first')
    parser.add_argument('--temporal_smooth', action='store_true',
                        help='启用时序一致性平滑 (方案D: 对背景区域做帧间平滑，缓解背景闪烁)')
    parser.add_argument("--smooth_weight", type=float, default=0.7,
                        help='时序平滑中历史帧权重 (默认0.7，越大背景越平滑但响应越慢)')

    parser.add_argument("--need_crop_source_img", action='store_true', help='crop input source_img. Please download shape_predictor_68_face_landmarks.dat and put it in ./data_preprocess first')
    parser.add_argument("--need_crop_pose_video", action='store_true', help='crop input pose_driving video.')


    args = parser.parse_args()

    demo = Demo(args)
    demo.run()


