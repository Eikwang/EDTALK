import os
import glob
from argparse import ArgumentParser
from moviepy import VideoFileClip

def split_video(input_file, duration=5, name=None, save_dir=None):

    video = VideoFileClip(input_file)
    total_duration = video.duration

    start_time = 0
    end_time = duration
    count = 1

    os.makedirs(save_dir, exist_ok=True)

    while start_time < total_duration:
        if end_time > total_duration:
            end_time = total_duration
        nnn = name+'#'+str(count)
        output_file = os.path.join(save_dir, nnn+'.mp4')
        sub_video = video.subclipped(start_time, end_time)
        sub_video.write_videofile(output_file)

        start_time += duration
        end_time += duration
        count += 1

    video.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Base directory path (contains video folder)")
    parser.add_argument("--duration", type=int, default=5, help="Split duration in seconds")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")

    args = parser.parse_args()

    root_dir = os.path.join(args.base_dir, 'video')
    save_dir = os.path.join(args.base_dir, f'split_{args.duration}s_video')
    os.makedirs(save_dir, exist_ok=True)

    videos = glob.glob1(root_dir, '*.mp4')
    print(f"Found {len(videos)} videos in {root_dir}")

    for v in videos:
        video_path = os.path.join(root_dir, v)
        split_video(video_path, args.duration, v.split('.')[0], save_dir)
