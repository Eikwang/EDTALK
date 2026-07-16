r"""
WDDM 主动重启包装器.

WDDM 驱动在长时间 CUDA 训练后虚拟内存映射退化, 导致 access violation
(C++ abort, Python 无法捕获). 解决方案: 每 N 步主动重启进程, 获取全新 CUDA 上下文.

用法:
  python run_fine_tune.py --datapath HDTF --resume_ckpt ckpts/EDTalk_lip_pose.pt --exp_path ./ckpt_models --exp_name wy --iter 50000 --batch_size 8 --lr 0.001

可选参数:
  --restart_every N    每 N 步主动重启 (默认 3000, 0=仅崩溃时重启)
"""
import argparse
import subprocess
import sys
import os
import glob
import time


def _parse_iter_from_name(name):
    """从 checkpoint 文件名解析迭代号, 返回 None 表示无法解析。"""
    if name.startswith("emergency_"):
        suffix = name[len("emergency_"):]
        try:
            return int(suffix)
        except ValueError:
            return None
    try:
        return int(name)
    except ValueError:
        return None


def find_latest_checkpoint(checkpoint_dir, min_iter=0):
    """在 checkpoint 目录中查找迭代编号最大的 .pt 文件。

    min_iter: 只返回迭代号 >= min_iter 的 checkpoint, 避免误用旧 checkpoint。
    """
    if not os.path.isdir(checkpoint_dir):
        return None, -1

    best_iter = -1
    best_file = None

    for f in glob.glob(os.path.join(checkpoint_dir, "*.pt")):
        name = os.path.splitext(os.path.basename(f))[0]
        n = _parse_iter_from_name(name)
        if n is not None and n >= min_iter and n > best_iter:
            best_iter = n
            best_file = f

    return best_file, best_iter


def main():
    # 解析需要拦截的关键参数, 其余原样透传给 train_fine_tune.py
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--exp_path", type=str, default=".")
    parser.add_argument("--exp_name", type=str, default="Obama")
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--restart_every", type=int, default=3000,
                        help="每 N 步主动重启进程以重置 WDDM 状态 (0=仅崩溃时重启)")
    parser.add_argument("-h", "--help", action="store_true")
    opts, passthrough = parser.parse_known_args()

    if opts.help:
        subprocess.run([sys.executable, "train_fine_tune.py", "--help"])
        return

    # 计算目标结束迭代
    if opts.resume_ckpt is not None:
        ckpt_name = os.path.basename(opts.resume_ckpt)
        try:
            initial_start = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            initial_start = opts.start_iter
    else:
        initial_start = 0

    target_end = initial_start + opts.iter
    checkpoint_dir = os.path.join(opts.exp_path, opts.exp_name, "checkpoint")

    # 确定有效的起始迭代: 取 resume_ckpt 迭代号和 checkpoint 目录最新文件的最大值
    # 这样即使用户不指定 --resume_ckpt, 也能从上次中断处继续
    _, dir_latest = find_latest_checkpoint(checkpoint_dir)
    effective_start = max(initial_start, dir_latest) if dir_latest > 0 else initial_start

    # 如果 checkpoint 目录已有进度超过 target_end, 询问用户
    if effective_start >= target_end:
        print("=" * 70)
        print("[WARNING] Checkpoint directory already has progress beyond target:")
        print("  Latest checkpoint in dir : iter %d" % dir_latest)
        print("  Resume checkpoint        : iter %d" % initial_start)
        print("  Target end iteration     : iter %d" % target_end)
        print()
        print("  Possible causes:")
        print("  - Previous training already completed this target")
        print("  - Need a larger --iter value to continue training")
        print("  - Need a different --exp_name for a new training run")
        print("=" * 70)
        return

    # 如果 checkpoint 目录有比 resume_ckpt 更新的进度, 自动使用
    if dir_latest > initial_start:
        latest_file, _ = find_latest_checkpoint(checkpoint_dir)
        print("[INFO] Checkpoint dir has newer progress (iter %d) than resume_ckpt (iter %d)." % (dir_latest, initial_start))
        print("[INFO] Will resume from: %s" % latest_file)
        initial_start = dir_latest
        opts.resume_ckpt = latest_file

    target_end = initial_start + opts.iter

    print("=" * 70)
    print("WDDM Auto-Restart Wrapper")
    print("  Restart interval : every %d iters" % opts.restart_every)
    print("-" * 70)
    print("  Start iteration  : %d" % initial_start)
    print("  Target end       : %d" % target_end)
    print("  Total iterations : %d" % opts.iter)
    print("  Checkpoint dir   : %s" % checkpoint_dir)
    print("=" * 70)

    current_resume = opts.resume_ckpt
    current_start = initial_start
    restart_count = 0
    total_elapsed = 0.0
    total_lost_iters = 0

    while True:
        remaining = target_end - current_start
        if remaining <= 0:
            print("\n" + "=" * 70)
            print("Training completed successfully!")
            print("  Total wall time : %.0fs (%.1fh)" % (total_elapsed, total_elapsed / 3600))
            print("  Restarts        : %d" % restart_count)
            print("  Lost iters      : %d" % total_lost_iters)
            print("=" * 70)
            return

        # 本轮运行的迭代数
        if opts.restart_every > 0:
            run_iter = min(opts.restart_every, remaining)
        else:
            run_iter = remaining

        # 构建命令
        cmd = [sys.executable, "train_fine_tune.py"]
        cmd.extend(passthrough)
        cmd.extend([
            "--iter", str(run_iter),
            "--exp_path", opts.exp_path,
            "--exp_name", opts.exp_name,
        ])
        if current_resume is not None:
            cmd.extend(["--resume_ckpt", current_resume])

        print("\n" + "=" * 70)
        if restart_count == 0:
            print("Starting training (iter %d -> %d, %d iters this run)"
                  % (current_start, current_start + run_iter, run_iter))
        else:
            print("Restart #%d: iter %d -> %d, %d iters this run"
                  % (restart_count, current_start, current_start + run_iter, run_iter))
        print("=" * 70 + "\n")

        start_time = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start_time
        total_elapsed += elapsed

        if result.returncode == 0:
            print("\n[NORMAL] Exited normally after %.0fs" % elapsed)
        else:
            print("\n[CRASH] Exit code %d after %.0fs" % (result.returncode, elapsed))
            print("[CRASH] Waiting 5s for GPU resources to release...")
            time.sleep(5)

        # 查找本轮产生的最新 checkpoint (只看 >= current_start 的, 避免误用旧 checkpoint)
        latest_ckpt, latest_iter = find_latest_checkpoint(checkpoint_dir, min_iter=current_start)

        if latest_ckpt is None or latest_iter < 0:
            print("[ERROR] No new checkpoint found (iter >= %d). Cannot restart." % current_start)
            sys.exit(1)

        if latest_iter >= target_end - 1:
            print("\n" + "=" * 70)
            print("Training completed successfully!")
            print("  Latest checkpoint   : %s (iter %d)" % (latest_ckpt, latest_iter))
            print("  Total wall time     : %.0fs (%.1fh)" % (total_elapsed, total_elapsed / 3600))
            print("  Restarts            : %d" % restart_count)
            print("  Lost iters          : %d" % total_lost_iters)
            print("=" * 70)
            return

        # 估算丢失的迭代数
        expected_end = current_start + run_iter
        lost = max(0, expected_end - latest_iter)
        lost = min(lost, run_iter)
        if lost > 0:
            total_lost_iters += lost
            print("[RESTART] Lost ~%d iters (expected %d, checkpoint at %d)"
                  % (lost, expected_end, latest_iter))
        else:
            print("[RESTART] No iters lost (checkpoint at %d)" % latest_iter)

        current_resume = latest_ckpt
        current_start = latest_iter
        restart_count += 1

        if restart_count > 50:
            print("[ERROR] Too many restarts (>50). Aborting.")
            sys.exit(1)


if __name__ == "__main__":
    main()
