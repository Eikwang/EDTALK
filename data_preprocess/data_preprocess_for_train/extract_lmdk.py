# -*- coding: utf-8 -*-
"""
从 LMDB 中提取每帧的 68 点面部关键点（dlib）
用法: python extract_lmdk.py --base_dir HDTF
"""

import os
import lmdb
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import dlib
from imutils import face_utils
from tqdm import tqdm
from argparse import ArgumentParser
import multiprocessing as mp

warnings_imported = False
try:
    import warnings
    warnings.filterwarnings("ignore")
    warnings_imported = True
except ImportError:
    pass


# ============================================================
# 全局变量（模块级，供多进程使用）
# ============================================================
_predictor_path = None
_detector = None
_predictor = None


def init_landmark_detector(predictor_path):
    """初始化 dlib 检测器和预测器（每个子进程调用一次）"""
    global _detector, _predictor, _predictor_path
    if _detector is not None:
        return
    _predictor_path = predictor_path
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(predictor_path)


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def get_video_list_from_lmdb(lmdb_path):
    """从 LMDB 中读取所有视频名列表"""
    env = lmdb.open(
        lmdb_path,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    if not env:
        raise IOError(f'Cannot open lmdb dataset: {lmdb_path}')

    videos = set()
    with env.begin(write=False) as txn:
        # 先获取总长度
        length_key = format_for_lmdb('length')
        total_bytes = txn.get(length_key)
        if total_bytes:
            total = int(total_bytes.decode('utf-8'))
        else:
            total = 0

        # 遍历所有 key，提取视频名
        cursor = txn.cursor()
        for key, _ in cursor:
            key_str = key.decode('utf-8')
            # key 格式: "0000005#71-0000003" 或 "0000005#71-length"
            # 去掉末尾的 "-数字部分" 或 "-length"
            if '-length' in key_str:
                video_name = key_str.rsplit('-length', 1)[0]
            else:
                # 取最后一个 "-" 之前的部分
                parts = key_str.rsplit('-', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    video_name = parts[0]
                else:
                    continue

            # 去掉 zfill 的前导零
            if video_name:
                videos.add(video_name)

    env.close()

    result = sorted(videos)

    # 如果通过遍历没找到，用总长度回退
    if len(result) == 0 and total > 0:
        print(f"[WARN] 无法从 LMDB key 解析视频名，尝试按索引查找 (total={total})")
        return [str(i) for i in range(total)]

    return result


def extract_landmarks_for_video(args_tuple):
    """处理单个视频：从 LMDB 读取所有帧 → 检测关键点 → 保存 .npy"""
    video_name, lmdb_path, save_dir, predictor_path = args_tuple

    # 确保每个进程初始化自己的检测器
    init_landmark_detector(predictor_path)

    env = lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    try:
        with env.begin(write=False) as txn:
            key = format_for_lmdb(video_name, 'length')
            length_bytes = txn.get(key)
            if length_bytes is None:
                print(f"[WARN] Video not found in LMDB: {video_name}")
                return False

            length = int(length_bytes.decode('utf-8'))
            landmarks = []

            for j in range(length):
                key = format_for_lmdb(video_name, j)
                img_bytes = txn.get(key)
                if img_bytes is None:
                    continue

                img = Image.open(BytesIO(img_bytes))
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

                rects = _detector(gray, 0)
                for rect in rects:
                    shape = _predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    landmarks.append(shape)

            if len(landmarks) == 0:
                print(f"[WARN] No face detected in any frame: {video_name}")
                return False

            landmarks = np.array(landmarks)
            save_path = os.path.join(save_dir, video_name + '.npy')
            np.save(save_path, landmarks)
            return True

    except Exception as e:
        print(f"[ERROR] {video_name}: {e}")
        return False
    finally:
        env.close()


def find_lmdb_path(base_dir):
    """自动查找 LMDB 目录"""
    lmdb_path = os.path.join(base_dir, 'lmdb')
    data_mdb = os.path.join(lmdb_path, 'data.mdb')
    if not os.path.exists(data_mdb):
        raise FileNotFoundError(
            f"未找到 LMDB 数据库，请先运行 prepare_lmdb.py。"
            f"\n  期望路径: {data_mdb}"
        )
    return lmdb_path


def find_predictor_path():
    """查找 shape_predictor_68_face_landmarks.dat"""
    # 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(project_root, 'ckpts', 'shape_predictor_68_face_landmarks.dat'),
        'shape_predictor_68_face_landmarks.dat',
        os.path.join(project_root, 'shape_predictor_68_face_landmarks.dat'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "未找到 shape_predictor_68_face_landmarks.dat。\n"
        f"期望路径: {candidates[0]}\n"
        "请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载。"
    )


def main(base_dir, num_processes=10):
    """
    从 LMDB 提取面部关键点

    用法: python extract_lmdk.py --base_dir HDTF
    """

    # 自动查找 LMDB 路径
    lmdb_path = find_lmdb_path(base_dir)

    # 输出目录：{base_dir}/landmark
    save_dir = os.path.join(base_dir, 'landmark')
    os.makedirs(save_dir, exist_ok=True)

    # 查找预测器模型
    predictor_path = find_predictor_path()

    print(f"LMDB   : {lmdb_path}")
    print(f"Output : {save_dir}")
    print(f"Model  : {predictor_path}")

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

    # 多进程提取
    mp.set_start_method('spawn', force=True)

    args_list = [(v, lmdb_path, save_dir, predictor_path) for v in pending]

    success = 0
    fail = 0

    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(extract_landmarks_for_video, args_list),
                          total=len(pending), desc="Extracting landmarks"):
            if result:
                success += 1
            else:
                fail += 1

    print(f"Done: {success} processed, {fail} failed")


if __name__ == "__main__":
    parser = ArgumentParser(description="从 LMDB 提取 68 点面部关键点")
    parser.add_argument("--base_dir", type=str, required=True, help="数据集主目录（如 HDTF）")
    parser.add_argument("--num_processes", type=int, default=10, help="工作进程数")
    args = parser.parse_args()

    main(args.base_dir, args.num_processes)
