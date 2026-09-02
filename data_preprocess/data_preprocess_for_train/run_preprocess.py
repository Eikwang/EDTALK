# -*- coding: utf-8 -*-
"""
EDTalk 数据集一键预处理脚本

按照 数据集处理流程.md 中的步骤，依次调用各子脚本完成全流程预处理：
  1.  crop_video_HDTF.py           - 面部裁剪            -> video/
  2.  split_HDTF_video.py          - 视频分割(5s)        -> split_5s_video/
  3.  split_HDTF_audio.py          - 音频分割(5s)        -> split_5s_audio/
  4.  prepare_lmdb.py              - 视频转 LMDB         -> lmdb/
  5.  get_mel.py                   - 梅尔谱图提取        -> mel/
  6.  extract_lmdk.py              - 关键点提取          -> landmark/
  7.  extract_bbox.py              - 边界框提取          -> bbox/
  8.  extract_lip_pose_features.py - 唇部姿态特征提取    -> lip_feature/, pose_feature/
  9.  prepare_feature_lmdb.py      - 特征转 LMDB         -> lmdb_features/
  10. crop_head_video.py           - 拆帧+面部裁剪       -> orifra/, count/, metadata/, facemark/

说明:
  - 通过 subprocess 依次调用各脚本（而非 import），避免多进程 start_method 冲突
    和模块级副作用；每个脚本运行在独立 Python 进程中。
  - 各子脚本本身支持幂等（跳过已处理文件），可安全重复运行。
  - 某步失败默认停止后续步骤，可用 --no-stop_on_fail 调整。

用法:
  # 完整流程
  python data_preprocess/data_preprocess_for_train/run_preprocess.py --base_dir HDTF

  # 只执行指定步骤
  python data_preprocess/data_preprocess_for_train/run_preprocess.py --base_dir HDTF --steps 1,2,3

  # 跳过指定步骤
  python data_preprocess/data_preprocess_for_train/run_preprocess.py --base_dir HDTF --skip_steps 6,7

  # 强制 CPU
  python data_preprocess/data_preprocess_for_train/run_preprocess.py --base_dir HDTF --device cpu
"""

import os
import sys
import time
import argparse
import subprocess

# 脚本所在目录: data_preprocess/data_preprocess_for_train/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


# 步骤定义: (序号, 名称, 脚本文件名, 输出目录说明)
PIPELINE_STEPS = [
    (1,  "面部裁剪",            "crop_video_HDTF.py",                 "video/"),
    (2,  "视频分割(5s)",        "split_HDTF_video.py",                "split_5s_video/"),
    (3,  "音频分割(5s)",        "split_HDTF_audio.py",                "split_5s_audio/"),
    (4,  "视频转LMDB",          "prepare_lmdb.py",                    "lmdb/"),
    (5,  "梅尔谱图提取",        "get_mel.py",                         "mel/"),
    (6,  "关键点提取",          "extract_lmdk.py",                    "landmark/"),
    (7,  "边界框提取",          "extract_bbox.py",                    "bbox/"),
    (8,  "唇部姿态特征提取",    "extract_lip_pose_features.py",       "lip_feature/, pose_feature/"),
    (9,  "特征转LMDB",          "prepare_feature_lmdb.py",            "lmdb_features/"),
    (10, "拆帧+面部裁剪",       "crop_head_video.py",                 "orifra/, count/, metadata/, facemark/"),
]


def resolve_script(filename):
    """解析脚本绝对路径，优先 SCRIPT_DIR，回退到项目根"""
    p = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(p):
        return p
    p2 = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(p2):
        return p2
    return p  # 返回默认路径，让后续报错更直观


def run_step(step_idx, name, script_file, base_dir, step_specific_args, dry_run=False):
    """执行单个步骤：以子进程方式调用对应脚本"""
    script_path = resolve_script(script_file)
    if not os.path.exists(script_path):
        print(f"[ERROR] 步骤 {step_idx} 脚本不存在: {script_path}")
        return False

    # -u: 强制子进程无缓冲输出，避免 tqdm/print 在非 TTY 环境下块缓冲
    # 导致“看不到进度”的假死现象（Windows 上从 IDE/重定向运行时尤其常见）
    cmd = [sys.executable, "-u", script_path]

    # crop_head_video.py 使用 positional input 而非 --base_dir
    if step_idx == 10:
        cmd.append(base_dir)
    else:
        cmd.extend(["--base_dir", base_dir])

    # 步骤特定参数
    cmd.extend(step_specific_args.get(step_idx, []))

    print("=" * 70, flush=True)
    print(f"[Step {step_idx}/10] {name}", flush=True)
    print(f"  脚本: {script_file}", flush=True)
    print(f"  命令: {' '.join(cmd)}", flush=True)
    print("=" * 70, flush=True)

    if dry_run:
        print("  (dry-run, 跳过实际执行)")
        return True

    # 子进程环境：强制无缓冲 + UTF-8，确保 tqdm/print 实时刷新
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # OpenMP 重复运行时修复(cv2+torch) + WDDM 显存碎片化缓解
    # 即使子脚本自身也设置了，这里再兜底一层，覆盖未设置的脚本(extract_lmdk 等)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"[FAIL] 步骤 {step_idx} 失败 (返回码 {proc.returncode}, 用时 {elapsed:.1f}s)")
            return False
        print(f"[OK] 步骤 {step_idx} 完成 (用时 {elapsed:.1f}s)")
        return True
    except KeyboardInterrupt:
        print(f"\n[中断] 步骤 {step_idx} 被用户中断")
        raise
    except Exception as e:
        print(f"[ERROR] 步骤 {step_idx} 异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="EDTalk 数据集一键预处理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_dir", required=True,
                        help="数据集主目录(需包含 original_videos/ 子目录)")
    parser.add_argument("--duration", type=int, default=5,
                        help="音视频分割时长(秒)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="人脸检测/GPU 推理设备")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID(用于唇部姿态特征提取)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="多进程/线程工作数")
    parser.add_argument("--steps", type=str, default=None,
                        help="只执行指定步骤,逗号分隔(如 1,2,3)。默认全部")
    parser.add_argument("--skip_steps", type=str, default=None,
                        help="跳过指定步骤,逗号分隔(如 6,7)。优先级高于 --steps")
    parser.add_argument("--image_shape", type=str, default="256,256",
                        help="crop_video_HDTF 输出图像尺寸 W,H")
    parser.add_argument("--output_size", type=int, default=256,
                        help="crop_head_video 输出图像边长")
    parser.add_argument("--output_format", type=str, default="jpg", choices=["jpg", "png"],
                        help="crop_head_video 输出图像格式")
    parser.add_argument("--longest_edge", type=int, default=1920,
                        help="crop_head_video 拆帧最长边")
    parser.add_argument("--no_landmark", action="store_true",
                        help="crop_head_video 禁用关键点检测")
    parser.add_argument("--no_stop_on_fail", dest="stop_on_fail", action="store_false",
                        help="某步失败时继续执行后续步骤(默认失败即停止)")
    parser.set_defaults(stop_on_fail=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令不实际执行")
    # Windows WDDM 安全参数：防止人脸检测长时间运行后 CUDA 死锁
    parser.add_argument("--safe_mode", dest="safe_mode", action="store_true",
                        help="Windows 安全模式 (等效于 --num_workers 1 + --detect_interval 3 + --no_landmark)")
    parser.add_argument("--detect_interval", type=int, default=1,
                        help="S3FD 人脸检测间隔 (帧): 1=每帧检测(最准最慢), 3=每3帧一次(快很多)")
    parser.add_argument("--detect_timeout_s", type=float, default=15.0,
                        help="单帧人脸检测超时阈值 (秒): Windows WDDM 下 S3FD 偶发死锁时跳帧")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="单步失败自动重试次数。配合步骤1的断点续跑，C级崩溃后可从断点继续")
    args = parser.parse_args()

    # Windows spawn 致命缺陷：子进程 re-import main 模块时 re-import torch，
    # 在 Py3.12 下触发 _lock_unlock_module SystemError，worker 崩溃后 Pool 永久等待。
    # 受影响步骤: 1(crop_video_HDTF.py)、6(extract_lmdk.py)、7(extract_bbox.py)
    #             均使用 spawn Pool 且模块级 import torch；
    #             步骤 10(crop_head_video.py) 是 I/O 写入线程池，不受影响。
    IS_WINDOWS = sys.platform.startswith("win")
    if IS_WINDOWS and args.num_workers > 1:
        print(
            f"[WARN] Windows + num_workers={args.num_workers}>1: 步骤 1/6/7 使用 spawn "
            "Pool 有极高概率触发 PyTorch 导入崩溃导致永久假死。"
        )
        if not args.safe_mode:
            print("[WARN] 建议加 --safe_mode (推荐) 或手工传 --num_workers 1")
    # safe_mode: 任何平台都强制串行
    if args.safe_mode:
        args.num_workers = 1
        if args.detect_interval < 3:
            args.detect_interval = 3
        args.no_landmark = True
        print("[SAFE_MODE] 已启用: num_workers=1(全流程无 spawn Pool), detect_interval=3, no_landmark")
    elif IS_WINDOWS and args.num_workers > 1:
        # 即使未启用 safe_mode，Windows 下依旧把可能崩溃的 Pool 步强制降到 1，
        # 避免用户卡死。受影响步骤: 1(crop_video_HDTF.py)、6(extract_lmdk.py)、
        # 7(extract_bbox.py)；步骤 4/10 为线程池不受影响。
        print("[WARN] Windows 下仍使用 num_workers>1，将把 spawn Pool 步骤(1,6,7)钳为 1 以规避崩溃")

    base_dir = os.path.abspath(args.base_dir)

    # 校验输入目录
    original_videos = os.path.join(base_dir, "original_videos")
    if not os.path.isdir(original_videos):
        print(f"[ERROR] 未找到 original_videos 目录: {original_videos}")
        print("        请将视频放入 <base_dir>/original_videos/ 后重试")
        sys.exit(1)

    # 解析步骤选择
    if args.steps:
        try:
            selected = {int(s) for s in args.steps.split(",") if s.strip()}
        except ValueError:
            print("[ERROR] --steps 参数格式错误,应为逗号分隔的数字(如 1,2,3)")
            sys.exit(1)
    else:
        selected = {s[0] for s in PIPELINE_STEPS}
    if args.skip_steps:
        try:
            skipped = {int(s) for s in args.skip_steps.split(",") if s.strip()}
        except ValueError:
            print("[ERROR] --skip_steps 参数格式错误,应为逗号分隔的数字(如 6,7)")
            sys.exit(1)
        selected -= skipped

    # 各步骤的特定参数
    # Windows 强制禁用 spawn Pool 的步骤：1 / 6 / 7（内部显式 spawn + Pool + 模块级 import torch）
    # 仅对 spawn Pool 步骤钳制，线程池步骤(4/10)不受影响
    spawn_pool_workers = 1 if IS_WINDOWS else args.num_workers
    step_args = {
        1: [
            "--image_shape", args.image_shape,
            "--detect_interval", str(args.detect_interval),
            "--detect_timeout_s", str(args.detect_timeout_s),
            "--num_processes", str(spawn_pool_workers),
        ],
        2: ["--duration", str(args.duration)],
        3: ["--duration", str(args.duration)],
        4: ["--n_worker", str(args.num_workers)],
        # extract_lmdk.py: --num_processes -> mp.Pool, Windows 强制 1
        6: ["--num_processes", str(spawn_pool_workers)],
        # extract_bbox.py: --num_workers -> mp.Pool, Windows 强制 1
        7: ["--num_workers", str(spawn_pool_workers)],
        8: ["--gpu_id", str(args.gpu_id)],
        10: [
            "--device", args.device,
            "--num_workers", str(args.num_workers),
            "--output_size", str(args.output_size),
            "--output_format", args.output_format,
            "--longest_edge", str(args.longest_edge),
        ],
    }
    if args.no_landmark:
        step_args[10].append("--no-landmark")
    if args.device == "cpu":
        step_args[1].append("--cpu")

    # 总览
    print("#" * 70)
    print("# EDTalk 数据集一键预处理")
    print("#" * 70)
    print(f"  base_dir   : {base_dir}")
    print(f"  duration   : {args.duration}s")
    print(f"  device     : {args.device}")
    print(f"  workers    : {args.num_workers}")
    print(f"  dry_run    : {args.dry_run}")
    print()
    print("  执行计划:")
    for idx, name, script, output in PIPELINE_STEPS:
        mark = "v" if idx in selected else "x"
        print(f"    [{mark}] {idx:2d}. {name:<20s} -> {output}")
    print()

    t_start = time.time()
    results = {}

    for idx, name, script, output in PIPELINE_STEPS:
        if idx not in selected:
            print(f"[SKIP] 步骤 {idx}. {name}")
            results[idx] = "skipped"
            continue

        # 自动重试: Windows 下 C 级崩溃(access violation)会静默杀掉子进程,
        # 步骤 1 有断点续跑, 重试从断点继续而不是从头再来
        ok = False
        for attempt in range(args.max_retries + 1):
            if attempt > 0:
                wait_s = min(3 * attempt, 15)
                print(f"\n[RETRY] 步骤 {idx} 第 {attempt}/{args.max_retries} 次重试 "
                      f"(等待 {wait_s}s 后开始, 断点续跑会从上次进度继续)...")
                time.sleep(wait_s)
            ok = run_step(idx, name, script, base_dir, step_args, dry_run=args.dry_run)
            if ok or args.dry_run:
                break
        results[idx] = "ok" if ok else "fail"

        if not ok and args.stop_on_fail:
            print(f"\n[STOP] 步骤 {idx} 重试 {args.max_retries} 次后仍失败,停止后续步骤")
            break

    # 总结
    elapsed = time.time() - t_start
    print()
    print("#" * 70)
    print(f"# 流程结束 (总用时 {elapsed:.1f}s)")
    print("#" * 70)
    for idx, name, _, output in PIPELINE_STEPS:
        status = results.get(idx, "n/a")
        symbol = {"ok": "OK", "fail": "FAIL", "skipped": "SKIP"}.get(status, status)
        print(f"  [{symbol:>4s}] {idx:2d}. {name:<20s} -> {output}")

    # 全部成功时输出最终目录结构
    if all(v in ("ok", "skipped") for v in results.values()) and not args.dry_run:
        print()
        print("数据集预处理完成! 目录结构:")
        print(f"  {base_dir}/")
        print("    original_videos/   原始视频(输入)")
        print("    video/             面部裁剪视频")
        print("    split_5s_video/    5秒视频片段")
        print("    split_5s_audio/    5秒音频片段")
        print("    lmdb/              视频帧 LMDB")
        print("    mel/               梅尔谱图")
        print("    landmark/          面部关键点")
        print("    bbox/              人脸边界框")
        print("    lip_feature/       唇部特征")
        print("    pose_feature/      姿态特征")
        print("    lmdb_features/     特征 LMDB")
        print("    orifra/            原始拆帧")
        print("    count/             面部裁剪帧")
        print("    metadata/          裁切元数据")
        print("    facemark/          面部关键点")

    sys.exit(0 if all(v in ("ok", "skipped") for v in results.values()) else 1)


if __name__ == "__main__":
    main()
