import os
import cv2
import lmdb
import argparse
import multiprocessing
import numpy as np
import glob as glob_module
from io import BytesIO
from tqdm import tqdm
from PIL import Image


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class HDTFResizer:
    """读取 HDTF 视频帧并编码为 JPEG bytes，用于写入 LMDB"""

    def __init__(self, video_dir, img_format='jpeg'):
        self.video_dir = video_dir
        self.img_format = img_format

    def prepare(self, filename):
        """读取单个视频的所有帧，返回 JPEG bytes 列表"""
        frames = []
        video_path = os.path.join(self.video_dir, filename + '.mp4')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open: {video_path}")
            return {'img': []}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame)
            img_bytes = self._get_bytes(img_pil)
            frames.append(img_bytes)
        cap.release()

        return {'img': frames}

    def _get_bytes(self, img):
        """直接编码为 JPEG bytes（视频已为 256×256，无需 resize）"""
        buf = BytesIO()
        img.save(buf, format=self.img_format)
        return buf.getvalue()

    def __call__(self, index_filename):
        index, filename = index_filename
        result = self.prepare(filename)
        return index, result, filename


def find_video_dir(base_dir):
    """自动查找 split_Xs_video 目录"""
    pattern = os.path.join(base_dir, 'split_*s_video')
    matches = sorted(glob_module.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"未找到 split_Xs_video 目录，请先运行 split_HDTF_video.py。"
            f"\n  搜索路径: {base_dir}/split_*s_video"
        )
    return matches[0]


def prepare_data(base_dir, out, n_worker=8, chunksize=10):
    """
    将 HDTF split 视频打包为 LMDB 数据库

    用法: python prepare_lmdb.py --base_dir HDTF --out EDTalk_lmdb
    """

    # 自动查找视频目录：HDTF/split_5s_video
    video_dir = find_video_dir(base_dir)

    # 扫描所有视频文件
    filenames = sorted([f.rsplit('.', 1)[0] for f in os.listdir(video_dir) if f.lower().endswith('.mp4')])
    total = len(filenames)

    print(f"Video dir : {video_dir}")
    print(f"Output    : {out}")
    print(f"Found {total} videos")

    os.makedirs(out, exist_ok=True)

    resizer = HDTFResizer(video_dir)

    with lmdb.open(out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            # 写入总长度
            txn.put(format_for_lmdb('length'), format_for_lmdb(total))

            with multiprocessing.Pool(n_worker) as pool:
                for idx, result, filename in tqdm(
                        pool.imap_unordered(resizer, enumerate(filenames), chunksize=chunksize),
                        total=total,
                        desc="Building LMDB"):
                    # 写入每个视频的帧数
                    txn.put(format_for_lmdb(filename, 'length'), format_for_lmdb(len(result['img'])))

                    # 写入每一帧
                    for frame_idx, frame in enumerate(result['img']):
                        txn.put(format_for_lmdb(filename, frame_idx), frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_dir', type=str, required=True, help='数据集主目录（如 HDTF）')
    parser.add_argument('--out', type=str, default=None, help='LMDB 输出目录（默认 {base_dir}/lmdb）')
    parser.add_argument('--n_worker', type=int, default=8, help='工作进程数')
    parser.add_argument('--chunksize', type=int, default=10, help='每个工作进程的分块大小')
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(args.base_dir, 'lmdb')

    prepare_data(**vars(args))
