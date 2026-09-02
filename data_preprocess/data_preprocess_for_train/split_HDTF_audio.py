import os
import warnings

# 抑制 pydub 的 SyntaxWarning 和 RuntimeWarning
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# 使用独立版 FFMPEG（更稳定，适合大规模数据处理）
# 优先使用 runtime312/ffmpeg/bin/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FFMPEG_BIN_DIR = os.path.join(PROJECT_ROOT, 'runtime312', 'ffmpeg', 'bin')

if os.path.exists(FFMPEG_BIN_DIR):
    # 将独立版 FFMPEG 添加到 PATH 最前面（优先级最高）
    os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ.get("PATH", "")
    print(f"[INFO] Using standalone FFMPEG: {FFMPEG_BIN_DIR}")
else:
    # 回退到 imageio-ffmpeg
    try:
        import imageio_ffmpeg
        os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ.get("PATH", "")
        print(f"[INFO] Using imageio-ffmpeg")
    except ImportError:
        print("[WARN] No FFMPEG found, using system default")

from pydub import AudioSegment
from moviepy import VideoFileClip
from argparse import ArgumentParser

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def extract_audio_from_videos(video_dir, audio_dir):
    """从 original_videos 目录提取音频到 audios 目录"""
    os.makedirs(audio_dir, exist_ok=True)

    video_files = sorted([f for f in os.listdir(video_dir)
                          if f.lower().endswith(VIDEO_EXTENSIONS)])
    print(f"Found {len(video_files)} video files in {video_dir}")

    for video_file in video_files:
        name = os.path.splitext(video_file)[0]
        audio_path = os.path.join(audio_dir, name + '.wav')

        if os.path.exists(audio_path):
            print(f"  Audio already exists: {audio_path}, skipping")
            continue

        video_path = os.path.join(video_dir, video_file)
        print(f"  Extracting audio from {video_file} ...")

        try:
            with VideoFileClip(video_path) as video:
                if video.audio is None:
                    print(f"  WARNING: No audio stream in {video_file}")
                    continue
                video.audio.write_audiofile(audio_path, logger=None)
        except Exception as e:
            print(f"  ERROR extracting audio from {video_file}: {e}")


def split(name, wav_dir, save_dir, duration=5):
    input_file = os.path.join(wav_dir, name+'.wav')

    audio = AudioSegment.from_wav(input_file)

    clip_length = duration * 1000
    clips = [audio[i:i+clip_length] for i in range(0, len(audio), clip_length)]

    os.makedirs(save_dir, exist_ok=True)
    for i, clip in enumerate(clips):
        filename = os.path.join(save_dir, f"{name}#{i+1}.wav")
        clip.export(filename, format="wav")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Base directory path (contains original_videos folder)")
    parser.add_argument("--duration", type=int, default=5, help="Split duration in seconds")

    args = parser.parse_args()

    # Step 1: 从 original_videos 提取音频到 audios
    video_dir = os.path.join(args.base_dir, 'original_videos')
    wav_dir = os.path.join(args.base_dir, 'audios')
    extract_audio_from_videos(video_dir, wav_dir)

    # Step 2: 从 audios 目录裁切音频
    save_dir = os.path.join(args.base_dir, f'split_{args.duration}s_audio')
    os.makedirs(save_dir, exist_ok=True)

    wav_list = sorted(os.listdir(wav_dir))
    print(f"Found {len(wav_list)} audio files in {wav_dir}")

    for name in wav_list:
        name = name.split('.')[0]
        print(name)
        split(name, wav_dir, save_dir, args.duration)
