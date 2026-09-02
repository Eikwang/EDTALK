"""
LMDB 数据完整性诊断脚本
用途：检查 LMDB 中不同帧索引是否真的对应不同的图像
"""
import lmdb
import numpy as np
from PIL import Image
from io import BytesIO
import hashlib
import os
import sys

# ============ 配置区 ============
LMDB_PATH = "HDTF/lmdb"           # ← 改成你的 LMDB 路径
SAMPLE_VIDEOS = 5                 # 抽查几个视频
FRAMES_PER_VIDEO = 50             # 每个视频抽查几帧
# ================================


def get_all_keys(env):
    """获取 LMDB 中所有 key"""
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        keys = []
        for key, _ in cursor:
            keys.append(key.decode('utf-8'))
        return keys


def parse_keys(keys):
    """
    解析 key 结构。
    你的训练脚本中 key 格式为: {video_name}-{frame_idx}
    其中 frame_idx 是 7 位零填充数字
    """
    from collections import defaultdict
    video_frames = defaultdict(list)
    
    for key in keys:
        # 尝试分离 video_name 和 frame_idx
        # key 格式示例: "2#110-0000050"
        last_dash = key.rfind('-')
        if last_dash == -1:
            print(f"  ⚠️  无法解析 key: {key}")
            continue
        video_name = key[:last_dash]
        frame_str = key[last_dash + 1:]
        
        try:
            frame_idx = int(frame_str)
            video_frames[video_name].append(frame_idx)
        except ValueError:
            print(f"  ⚠️  frame 部分不是数字: {key}")
            continue
    
    # 排序
    for v in video_frames:
        video_frames[v].sort()
    
    return dict(video_frames)


def load_image_bytes(env, key_str):
    """从 LMDB 读取原始图像字节"""
    with env.begin(write=False) as txn:
        data = txn.get(key_str.encode('utf-8'))
    return data


def image_to_array(img_bytes):
    """将图像字节转为 numpy array"""
    if img_bytes is None:
        return None
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def compute_image_hash(img_bytes):
    """计算图像字节的 MD5"""
    if img_bytes is None:
        return None
    return hashlib.md5(img_bytes).hexdigest()


def main():
    if not os.path.exists(LMDB_PATH):
        print(f"❌ LMDB 路径不存在: {LMDB_PATH}")
        sys.exit(1)

    print(f"📂 打开 LMDB: {LMDB_PATH}")
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    
    with env.begin(write=False) as txn:
        stats = txn.stat()
        total_entries = stats['entries']
        print(f"📊 LMDB 总条目数: {total_entries}")
    
    # ========== 第一步：解析 key 结构 ==========
    print(f"\n{'='*60}")
    print("📋 第一步：解析 key 结构")
    print(f"{'='*60}")
    
    all_keys = get_all_keys(env)
    print(f"  总 key 数: {len(all_keys)}")
    
    if len(all_keys) == 0:
        print("❌ LMDB 为空！")
        sys.exit(1)
    
    # 显示前 10 个 key 样本
    print(f"  前 10 个 key 样本:")
    for k in all_keys[:10]:
        print(f"    {k}")
    
    video_frames = parse_keys(all_keys)
    print(f"\n  检测到 {len(video_frames)} 个视频")
    
    if len(video_frames) == 0:
        print("❌ 无法解析任何 video-frame key！")
        print("  请检查 LMDB key 格式是否与预期一致")
        sys.exit(1)
    
    # 显示几个视频的帧数统计
    sorted_videos = sorted(video_frames.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\n  帧数最多的 5 个视频:")
    for vname, frames in sorted_videos[:5]:
        print(f"    {vname}: {len(frames)} 帧 (范围: {frames[0]}-{frames[-1]})")
    
    # ========== 第二步：逐字节比较相邻帧 ==========
    print(f"\n{'='*60}")
    print("🔍 第二步：检查不同帧是否为不同图像")
    print(f"{'='*60}")
    
    test_videos = [v for v, f in sorted_videos[:SAMPLE_VIDEOS] if len(f) >= FRAMES_PER_VIDEO]
    
    if not test_videos:
        test_videos = [v for v, f in sorted_videos if len(f) >= 2][:SAMPLE_VIDEOS]
        FRAMES_TO_CHECK = min(FRAMES_PER_VIDEO, min(len(video_frames[v]) for v in test_videos))
    else:
        FRAMES_TO_CHECK = FRAMES_PER_VIDEO
    
    total_pairs = 0
    identical_pairs = 0
    all_identical_count = 0  # 整个视频所有帧都一样的情况
    
    for vname in test_videos:
        frames = video_frames[vname]
        # 均匀采样
        if len(frames) > FRAMES_TO_CHECK:
            indices = np.linspace(0, len(frames) - 1, FRAMES_TO_CHECK, dtype=int)
            selected_frames = [frames[i] for i in indices]
        else:
            selected_frames = frames
        
        print(f"\n  📹 视频: {vname} (总帧数: {len(frames)}, 抽查: {len(selected_frames)})")
        
        hashes = []
        byte_sizes = []
        img_shapes = []
        
        for fidx in selected_frames:
            key_str = f"{vname}-{fidx:07d}"
            img_bytes = load_image_bytes(env, key_str)
            
            if img_bytes is None:
                print(f"    ⚠️  key 不存在: {key_str}")
                hashes.append(None)
                byte_sizes.append(0)
                img_shapes.append(None)
                continue
            
            h = compute_image_hash(img_bytes)
            hashes.append(h)
            byte_sizes.append(len(img_bytes))
            
            img_arr = image_to_array(img_bytes)
            img_shapes.append(img_arr.shape if img_arr is not None else None)
        
        # 比较相邻帧
        print(f"    {'帧索引':<12} {'MD5':<34} {'字节数':<10} {'形状'}")
        print(f"    {'-'*75}")
        
        video_all_same = True
        for i, fidx in enumerate(selected_frames):
            h = hashes[i] if i < len(hashes) else None
            bs = byte_sizes[i] if i < len(byte_sizes) else 0
            sh = img_shapes[i] if i < len(img_shapes) else None
            print(f"    {fidx:<12} {str(h):<34} {bs:<10} {sh}")
            
            if i > 0 and h is not None and hashes[i-1] is not None:
                total_pairs += 1
                if h == hashes[i-1]:
                    identical_pairs += 1
                    print(f"      ↑ ⚠️  与上一帧 MD5 完全相同！")
            
            if i > 0 and hashes[i] != hashes[0]:
                video_all_same = False
        
        if video_all_same and len(hashes) > 1 and hashes[0] is not None:
            all_identical_count += 1
            print(f"    🔴 严重：该视频所有 {len(selected_frames)} 帧图像完全相同！")
    
    # ========== 第三步：诊断总结 ==========
    print(f"\n{'='*60}")
    print("📊 诊断总结")
    print(f"{'='*60}")
    
    print(f"  检查的视频数: {len(test_videos)}")
    print(f"  检查的相邻帧对数: {total_pairs}")
    print(f"  其中完全相同的帧对: {identical_pairs}")
    print(f"  整视频所有帧都相同的视频数: {all_identical_count}/{len(test_videos)}")
    
    if total_pairs > 0:
        identical_ratio = identical_pairs / total_pairs
        if identical_ratio > 0.8:
            print(f"\n  🔴🔴🔴 严重问题：{identical_ratio*100:.1f}% 的相邻帧完全相同！")
            print(f"  这几乎确认了 LMDB 数据写入存在 bug。")
            print(f"  你的 SyncNet 训练无法收敛的根因就在这里——")
            print(f"  不同帧索引加载到的是同一张图，模型根本无法学到时序差异。")
        elif identical_ratio > 0.3:
            print(f"\n  🟡 警告：{identical_ratio*100:.1f}% 的相邻帧完全相同")
            print(f"  可能存在部分帧重复的问题，建议检查数据预处理流程。")
        elif identical_ratio > 0:
            print(f"\n  🟢 少量帧重复 ({identical_ratio*100:.1f}%)，可能是静态场景导致")
        else:
            print(f"\n  ✅ 所有帧都不同，LMDB 数据正常")
    
    # ========== 第四步：如果是 bug，给出修复方向 ==========
    if identical_pairs > 0 and total_pairs > 0 and identical_pairs / total_pairs > 0.3:
        print(f"\n{'='*60}")
        print("🔧 下一步：排查数据写入 bug")
        print(f"{'='*60}")
        print("""
  常见原因：
  1. 写入脚本中帧索引变量未正确递增
     → 检查 for 循环中 frame_idx 是否在外层被错误覆盖
  
  2. 所有帧写入了同一张图的引用（如只 decode 了第一帧）
     → 检查 cv2.VideoCapture.read() 是否在循环内正确调用
  
  3. 写入时 key 格式拼接错误，多个帧映射到同一个 key
     → 检查 f-string 中帧索引的格式化是否正确
  
  4. LMDB map_size 不够导致后续写入静默失败
     → 检查 lmdb.open() 的 map_size 参数

  建议：找到你的 LMDB 数据构建脚本，重点检查写入循环。
""")
    
    env.close()
    print("✅ 诊断完成")


if __name__ == "__main__":
    main()