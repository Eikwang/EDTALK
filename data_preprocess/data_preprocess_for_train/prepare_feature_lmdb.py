"""将 mel/lip_feature/pose_feature/bbox .npy 文件打包到 LMDB

适用于:
  - 大量数据集泛化训练 (HDTF 345+ 视频): 一次性预处理后训练加速
  - 少量数据集微调训练 (单人视频): 同样可预处理, 也可直接在内存中加载

用法:
  # HDTF 泛化训练
  python data_preprocess/data_preprocess_for_train/prepare_feature_lmdb.py --base_dir HDTF

  # 单人微调数据集
  python data_preprocess/data_preprocess_for_train/prepare_feature_lmdb.py --base_dir /path/to/finetune_data

  # 自定义特征子集
  python data_preprocess/data_preprocess_for_train/prepare_feature_lmdb.py --base_dir HDTF --features mel,lip_feature,pose_feature,bbox

生成目录: {base_dir}/lmdb_features/
后续训练时 Audio2LipDataset_image_sync 会自动检测并使用, 无需修改训练命令。
"""
import os
import lmdb
import argparse
import numpy as np
from io import BytesIO
from tqdm import tqdm


# 默认打包的特征列表 (与 dataset_audio2lip.py __getitem__ 读取一致)
DEFAULT_FEATURES = ['mel', 'lip_feature', 'pose_feature', 'bbox']


def _array_to_bytes(arr):
    """numpy 数组序列化为 bytes (保留 shape/dtype 元信息)

    使用 np.save 到 BytesIO, 自带 header 包含 shape 和 dtype,
    读取时用 np.load(BytesIO(data)) 即可完整恢复。
    """
    buf = BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def prepare_feature_lmdb(base_dir, out=None, feature_names=None):
    """将 mel/lip_feature/pose_feature/bbox .npy 打包到 LMDB

    Args:
        base_dir: 数据集主目录 (如 HDTF, 或微调数据目录)
                  需包含子目录: mel/, lip_feature/, pose_feature/, bbox/
        out: LMDB 输出目录 (默认 {base_dir}/lmdb_features)
        feature_names: 要打包的特征目录名列表

    LMDB 键格式:
        {feature_name}-{video_name}  ->  numpy array bytes
        __feature_names__            ->  "|" 分隔的特征名列表
        __video_count__             ->  视频总数
    """
    if out is None:
        out = os.path.join(base_dir, 'lmdb_features')
    if feature_names is None:
        feature_names = list(DEFAULT_FEATURES)

    # 以 mel 目录为基准收集视频名 (mel 是必需特征)
    mel_dir = os.path.join(base_dir, 'mel')
    if not os.path.isdir(mel_dir):
        raise FileNotFoundError(
            f"mel 目录不存在: {mel_dir}\n"
            f"请确认 base_dir 路径正确, 且已运行 get_mel.py 生成 mel 特征。"
        )

    video_names = sorted([
        f.rsplit('.', 1)[0]
        for f in os.listdir(mel_dir)
        if f.endswith('.npy')
    ])

    print(f"Base dir : {base_dir}")
    print(f"Output   : {out}")
    print(f"Features : {feature_names}")
    print(f"Videos   : {len(video_names)}")

    os.makedirs(out, exist_ok=True)

    # LMDB map_size: 1TB 足够 (特征总量远小于图像 LMDB)
    map_size = 1024 ** 4

    total_written = 0
    skipped = []

    with lmdb.open(out, map_size=map_size, readahead=False) as env:
        with env.begin(write=True) as txn:
            # 写入元数据
            txn.put(b'__feature_names__', '|'.join(feature_names).encode('utf-8'))
            txn.put(b'__video_count__', str(len(video_names)).encode('utf-8'))

            for name in tqdm(video_names, desc="Building feature LMDB"):
                for feat in feature_names:
                    npy_path = os.path.join(base_dir, feat, name + '.npy')
                    if not os.path.exists(npy_path):
                        skipped.append(f"{feat}/{name}.npy")
                        continue
                    try:
                        arr = np.load(npy_path)
                    except Exception as e:
                        print(f"[WARN] 无法加载 {npy_path}: {e}")
                        skipped.append(f"{feat}/{name}.npy (load error)")
                        continue

                    key = f"{feat}-{name}".encode('utf-8')
                    txn.put(key, _array_to_bytes(arr))
                    total_written += 1

    print(f"\nDone! 写入 {total_written} 个特征到 {out}")
    if skipped:
        print(f"[WARN] 跳过 {len(skipped)} 个缺失文件:")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... (共 {len(skipped)} 个)")

    # 估算总大小
    total_size = 0
    for dirpath, _, filenames in os.walk(out):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f"LMDB 总大小: {total_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="将 mel/lip_feature/pose_feature/bbox .npy 打包到 LMDB, "
                    "加速训练时的特征读取 IO。"
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='数据集主目录 (如 HDTF 或微调数据目录)')
    parser.add_argument('--out', type=str, default=None,
                        help='LMDB 输出目录 (默认 {base_dir}/lmdb_features)')
    parser.add_argument('--features', type=str, default=None,
                        help='要打包的特征 (逗号分隔, 默认 mel,lip_feature,pose_feature,bbox)')
    args = parser.parse_args()

    feature_names = None
    if args.features:
        feature_names = args.features.split(',')

    prepare_feature_lmdb(args.base_dir, args.out, feature_names)
