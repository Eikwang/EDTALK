import os

# 抑制 TensorFlow / oneDNN / absl 启动日志：被 torchvision 等依赖间接导入，
# 会在训练过程中反复打印 oneDNN port.cc 警告，污染进度输出。
# 必须在任何第三方 import (torch/torchvision) 之前设置，否则 TF C++ 层已初始化，日志已发出。
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')      # 仅保留 FATAL，屏蔽 INFO/WARN
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')     # 关闭 oneDNN 自定义算子提示

import argparse
import platform
import multiprocessing
import sys

# Windows 多进程: 必须使用 spawn 方式
if platform.system() == 'Windows':
    multiprocessing.set_start_method('spawn', force=True)

import torch
from torch.utils import data
from fine_tune.dataset import Finetune256
import torchvision
import torchvision.transforms as transforms
from fine_tune.trainer_fine_tune import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision import utils

torch.backends.cudnn.enabled = True
# cudnn.benchmark=True 会在首次前向时自动 benchmark 多个 kernel 选最快的,
# 但在 Windows + RTX 40 系 (Ada Lovelace, sm_89) + CUDA 12.8 下, 某些候选
# kernel 存在 bug, 触发 access violation (0xC0000005) 静默崩溃。
# 关闭 benchmark 使用 cudnn 默认的稳定 kernel, 速度差异很小 (<5%) 但更可靠。
torch.backends.cudnn.benchmark = False

# Windows WDDM 防护: 限制 CUDA 内存分配器, 减少虚拟内存碎片
if platform.system() == 'Windows':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6'


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def write_loss(i, vgg_loss, l1_loss, g_loss, d_loss, writer):
    writer.add_scalar('vgg_loss', vgg_loss.item(), i)
    writer.add_scalar('l1_loss', l1_loss.item(), i)
    writer.add_scalar('gen_loss', g_loss.item(), i)
    writer.add_scalar('dis_loss', d_loss.item(), i)
    writer.flush()


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    if args.distributed:
        ddp_setup(args, rank, world_size)
        torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    # 解析数据集目录: 在 datapath 下寻找 count 子目录
    count_dir = os.path.join(args.datapath, 'count')
    if not os.path.isdir(count_dir):
        count_dir = args.datapath
    print(f'  Using data directory: {count_dir}')

    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    dataset = Finetune256(count_dir, transform=transform,
                          same_identity_ratio=args.same_identity_ratio)

    if args.distributed:
        loader = data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=args.batch_size // world_size,
            sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
            pin_memory=True,
            drop_last=True,
        )
    else:
        loader = data.DataLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True if args.num_workers > 0 else False,
            drop_last=True,
        )

    loader = sample_data(loader)

    print('==> initializing trainer')
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    # WDDM 防护间隔: 每 N 步 synchronize + empty_cache, 让 WDDM 驱动回收资源
    # Windows 下 WDDM 驱动虚拟内存映射会逐步退化, 导致 C 级 access violation
    # 定期 synchronize 确保 GPU 完成 + empty_cache 释放碎片化显存
    wddm_sync_interval = args.wddm_sync_interval

    print('==> training', flush=True)
    pbar = range(args.iter)
    last_saved_iter = args.start_iter

    try:
        for idx in pbar:
            i = idx + args.start_iter

            # loading data
            img_source, img_target = next(loader)
            img_source = img_source.to(device, non_blocking=True)
            img_target = img_target.to(device, non_blocking=True)

            # update generator
            vgg_loss, l1_loss, gan_g_loss, img_recon = trainer.gen_update(img_source, img_target)

            # update discriminator
            gan_d_loss = trainer.dis_update(img_target, img_recon)

            # detach recon (free computation graph before display/val)
            img_recon = img_recon.detach()

            # WDDM 防护: 定期 synchronize + empty_cache 防止驱动状态退化
            if wddm_sync_interval > 0 and (idx + 1) % wddm_sync_interval == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # write to log
            if not args.distributed or rank == 0:
                write_loss(idx, vgg_loss, l1_loss, gan_g_loss, gan_d_loss, writer)

            # display
            if i % args.display_freq == 0 and (not args.distributed or rank == 0):
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print("[Iter %d/%d] [vgg: %f] [l1: %f] [g: %f] [d: %f] [mem: alloc=%.2fGB reserved=%.2fGB]"
                      % (i, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item(),
                         mem_alloc, mem_reserved), flush=True)

                # display 样本图: 从训练集随机抽样（覆盖所有视频）
                try:
                    with torch.no_grad():
                        img_test_source, img_test_target = next(loader)
                        img_test_source = img_test_source.to(device, non_blocking=True)
                        img_test_target = img_test_target.to(device, non_blocking=True)

                        img_recon_test = trainer.sample(img_test_source, img_test_target)

                        # 交错排列: 每行 [Source | Target | Recon] 一个样本，方便对比
                        B = img_test_source.size(0)
                        interleaved = torch.stack(
                            [img_test_source, img_test_target, img_recon_test], dim=1
                        ).reshape(B * 3, *img_test_source.shape[1:])
                        sample = F.interpolate(interleaved, 256)
                        utils.save_image(
                            sample,
                            os.path.join(checkpoint_path, "step_%05d.jpg" % (i)),
                            nrow=3,
                            normalize=True,
                            value_range=(-1, 1),
                        )
                        del img_test_source, img_test_target, img_recon_test, interleaved, sample
                    torch.cuda.synchronize()
                except Exception as e:
                    print("[WARN] display failed at iter %d: %s" % (i, str(e)), flush=True)

            # save model
            if i % args.save_freq == 0 and (not args.distributed or rank == 0):
                torch.cuda.synchronize()
                trainer.save(i, checkpoint_path)
                last_saved_iter = i

    except KeyboardInterrupt:
        print("\n==> Training interrupted by user at iter %d" % (idx + args.start_iter), flush=True)
    except Exception as e:
        print("\n==> Training crashed at iter %d: %s" % (idx + args.start_iter, str(e)), flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # 崩溃/中断时保存 checkpoint (不丢失训练进度)
        current_iter = (idx + args.start_iter) if 'idx' in dir() else args.start_iter
        if (not args.distributed or rank == 0) and current_iter > last_saved_iter:
            try:
                torch.cuda.synchronize()
                trainer.save(current_iter, checkpoint_path)
                print("==> Emergency checkpoint saved at iter %d." % current_iter, flush=True)
            except Exception as save_err:
                print("[ERROR] Failed to save emergency checkpoint: %s" % str(save_err), flush=True)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--only_fine_tune_dec", action='store_true',
                        help='仅微调 Decoder (冻结 Encoder+Direction+fc)')
    parser.add_argument("--freeze_direction", action='store_true',
                        help='冻结 Direction+fc, 训练 Encoder+Decoder (仅全量微调时生效)')
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default='ckpts/EDTalk_lip_pose.pt')
    parser.add_argument("--datapath", type=str, required=True,
                        help='Dataset directory (must contain count/ subdirectory with frame images)')
    parser.add_argument("--lr", type=float, default=0.001,
                        help='学习率 (默认 0.0001, 较低 lr 减少对预训练权重的扰动)')
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=5000,
                        help='检查点保存间隔 (步), 建议设为 2000-5000 以减少崩溃丢失')
    parser.add_argument("--wddm_sync_interval", type=int, default=500,
                        help='WDDM 防护: 每 N 步 synchronize+empty_cache (0=禁用, Windows 建议 100)')
    parser.add_argument("--exp_path", type=str, default='ckpt_models',
                        help='实验根目录 (保存 checkpoint 和 log; 默认 ckpt_models 即项目根目录)')
    parser.add_argument("--exp_name", type=str, default='Obama')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--num_workers", type=int, default=2,
                        help='DataLoader worker 进程数 (0=主线程加载)')
    parser.add_argument("--direction_lr_ratio", type=float, default=5.0,
                        help='Direction 模块学习率倍数 (无冻结训练时生效, 默认 5.0, 帮助 Direction 快速适应新数据)')
    parser.add_argument("--self_recon_weight", type=float, default=0.5,
                        help='自重建正则化权重 (默认 0.5, 防止 Decoder 跨身份训练时漂移; 0=禁用)')
    parser.add_argument("--same_identity_ratio", type=float, default=0.8,
                        help='同身份采样比例 (默认 0.8, 从同一视频选source/target帧; 0=全部跨身份)')

    # deprecated params (backward compat, ignored)
    parser.add_argument("--warmup_iters", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--val_freq", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--patience", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--temporal_window", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--far_window", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--far_ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--cross_window_min", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cross_window_max", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--id_weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--cross_weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lip_space_weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--pose_space_weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--beta1", type=float, default=0.0,
                        help='Adam beta1 (动量), 原始训练=0, 建议微调用 0 或 0.1 (默认 0.0)')
    parser.add_argument("--d_weight_decay", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--r1_batch_size", type=int, default=4, help=argparse.SUPPRESS)
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()

    if n_gpus >= 3:
        opts.distributed = True
        world_size = n_gpus
        print('==> training on %d gpus (distributed)' % n_gpus)
        mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)
    elif n_gpus == 1:
        opts.distributed = False
        print('==> training on 1 gpu (single)')
        main(0, 1, opts)
    else:
        raise RuntimeError("No GPU available")
