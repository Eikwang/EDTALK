# -*- coding: utf-8 -*-
"""
从 LMDB 中提取每帧的人脸 bbox（S3FD 检测，多进程并行版）
用法: python extract_bbox.py --base_dir HDTF
"""

import os
import sys
import lmdb
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ============================================================
# S3FD 面部检测器（单帧检测）
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def get_video_list_from_lmdb(lmdb_path):
    """从 LMDB 中读取所有视频名列表"""
    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        raise IOError(f'Cannot open lmdb dataset: {lmdb_path}')

    videos = set()
    with env.begin(write=False) as txn:
        length_key = format_for_lmdb('length')
        total_bytes = txn.get(length_key)
        total = int(total_bytes.decode('utf-8')) if total_bytes else 0

        cursor = txn.cursor()
        for key, _ in cursor:
            key_str = key.decode('utf-8')
            if '-length' in key_str:
                video_name = key_str.rsplit('-length', 1)[0]
            else:
                parts = key_str.rsplit('-', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    video_name = parts[0]
                else:
                    continue
            if video_name:
                videos.add(video_name)

    env.close()
    result = sorted(videos)

    if len(result) == 0 and total > 0:
        print(f"[WARN] 无法从 LMDB key 解析视频名，尝试按索引查找 (total={total})")
        return [str(i) for i in range(total)]

    return result


# ============================================================
# 全局检测器（每个子进程独立初始化）
# ============================================================
_detector = None


def init_detector():
    """在子进程中初始化检测器"""
    global _detector
    if _detector is not None:
        return
    from ckpts.s3fd import S3FD
    _detector = S3FD(device='cuda')


def detect_face_batch(frames, detector, conf_th=0.5):
    """逐帧检测人脸（无批量支持），返回 [x1,y1,x2,y2] 列表"""
    results = []
    for frame in frames:
        bboxes = detector.detect_faces(frame, conf_th=conf_th)
        if len(bboxes) > 0:
            # 选面积最大的
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best = int(np.argmax(areas))
            x1, y1, x2, y2 = bboxes[best][:4]
            results.append([int(x1), int(y1), int(x2), int(y2)])
        else:
            results.append(None)
    return results


def process_video(video_name, lmdb_path, save_dir):
    """处理单个视频"""
    global _detector
    save_path = os.path.join(save_dir, video_name + '.npy')

    if os.path.exists(save_path):
        return True, video_name

    # 延迟初始化检测器
    if _detector is None:
        init_detector()

    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

    try:
        frames = []

        with env.begin(write=False) as txn:
            key = format_for_lmdb(video_name, 'length')
            length_bytes = txn.get(key)
            if length_bytes is None:
                return False, video_name

            length = int(length_bytes.decode('utf-8'))

            for j in range(length):
                key = format_for_lmdb(video_name, j)
                img_bytes = txn.get(key)
                if img_bytes is None:
                    continue
                img = Image.open(BytesIO(img_bytes))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                frames.append(img)

        if len(frames) == 0:
            return False, video_name

        # 逐帧检测（无批量支持）
        bboxes = detect_face_batch(frames, _detector, conf_th=0.5)

        # 只保留有效检测的帧
        valid_bboxes = [b for b in bboxes if b is not None]

        if len(valid_bboxes) == 0:
            return False, video_name

        bbox_array = np.array(valid_bboxes)
        np.save(save_path, bbox_array)
        return True, video_name

    except Exception as e:
        print(f"[ERROR] {video_name}: {e}")
        return False, video_name
    finally:
        env.close()


def find_lmdb_path(base_dir):
    """自动查找 LMDB 目录"""
    lmdb_path = os.path.join(base_dir, 'lmdb')
    data_mdb = os.path.join(lmdb_path, 'data.mdb')
    if not os.path.exists(data_mdb):
        raise FileNotFoundError(
            f"未找到 LMDB 数据库，请先运行 prepare_lmdb.py。\n"
            f"期望路径: {data_mdb}"
        )
    return lmdb_path


def main(base_dir, num_workers=8):
    """
    从 LMDB 提取每帧的人脸 bbox（S3FD）

    用法: python extract_bbox.py --base_dir HDTF [--num_workers 4]
    """

    lmdb_path = find_lmdb_path(base_dir)
    save_dir = os.path.join(base_dir, 'bbox')
    os.makedirs(save_dir, exist_ok=True)

    print(f"LMDB      : {lmdb_path}")
    print(f"Output    : {save_dir}")
    print(f"Workers   : {num_workers}")

    # 从 LMDB 读取视频列表
    videos = get_video_list_from_lmdb(lmdb_path)
    print(f"Found {len(videos)} videos")

    # 过滤已处理的
    pending = [v for v in videos if not os.path.exists(os.path.join(save_dir, v + '.npy'))]
    if len(pending) < len(videos):
        print(f"Skip {len(videos) - len(pending)} already processed")

    if not pending:
        print("All done!")
        return

    # 多进程并行处理（每个进程独立初始化 S3FD）
    mp.set_start_method('spawn', force=True)

    process_func = partial(process_video, lmdb_path=lmdb_path, save_dir=save_dir)

    success = 0
    fail = 0

    with mp.Pool(processes=num_workers, initializer=init_detector) as pool:
        for ok, vname in tqdm(pool.imap_unordered(process_func, pending),
                              total=len(pending), desc="Extracting bbox"):
            if ok:
                success += 1
            else:
                fail += 1

    print(f"Done: {success} processed, {fail} failed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从 LMDB 提取人脸 bbox（S3FD）")
    parser.add_argument("--base_dir", type=str, required=True, help="数据集主目录（如 HDTF）")
    parser.add_argument("--num_workers", type=int, default=8, help="工作进程数")
    args = parser.parse_args()

    main(args.base_dir, args.num_workers)