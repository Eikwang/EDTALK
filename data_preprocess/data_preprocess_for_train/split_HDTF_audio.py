from pydub import AudioSegment
import os
from argparse import ArgumentParser

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
    parser.add_argument("--base_dir", required=True, help="Base directory path (contains crop_video folder)")
    parser.add_argument("--duration", type=int, default=5, help="Split duration in seconds")

    args = parser.parse_args()

    wav_dir = os.path.join(args.base_dir, 'audios')
    save_dir = os.path.join(args.base_dir, f'split_{args.duration}s_audio')
    os.makedirs(save_dir, exist_ok=True)

    wav_list = sorted(os.listdir(wav_dir))
    print(f"Found {len(wav_list)} audio files in {wav_dir}")

    for name in wav_list:
        name = name.split('.')[0]
        print(name)
        split(name, wav_dir, save_dir, args.duration)
