import argparse
import os
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
torch.backends.cudnn.benchmark = True

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

    dataset = Finetune256(count_dir, True, transform=transform)
    dataset_test = Finetune256(count_dir, False, transform=transform)

    if args.distributed:
        loader = data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=args.batch_size // world_size,
            sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
            pin_memory=True,
            drop_last=False,
        )
        loader_test = data.DataLoader(
            dataset_test,
            num_workers=4,
            batch_size=4,
            sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
            pin_memory=True,
            drop_last=False,
        )
    else:
        loader = data.DataLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True if args.num_workers > 0 else False,
            drop_last=False,
        )
        loader_test = data.DataLoader(
            dataset_test,
            num_workers=args.num_workers,
            batch_size=4,
            shuffle=False,
            pin_memory=True if args.num_workers > 0 else False,
            drop_last=False,
        )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

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

    print('==> training')
    pbar = range(args.iter)
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

            with torch.no_grad():
                img_test_source, img_test_target = next(loader_test)
                img_test_source = img_test_source.to(device, non_blocking=True)
                img_test_target = img_test_target.to(device, non_blocking=True)

                img_recon_test = trainer.sample(img_test_source, img_test_target)

                sample = F.interpolate(
                    torch.cat((img_test_source, img_test_target, img_recon_test), dim=0), 256)
                utils.save_image(
                    sample,
                    os.path.join(checkpoint_path, "step_%05d.jpg" % (i)),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                del img_test_source, img_test_target, img_recon_test, sample

            # display 后 synchronize 确保 GPU→CPU 传输完成
            torch.cuda.synchronize()

        # save model
        if i % args.save_freq == 0 and (not args.distributed or rank == 0):
            torch.cuda.synchronize()
            trainer.save(i, checkpoint_path)

    # 训练结束保存最终 checkpoint
    final_iter = args.start_iter + args.iter
    if not args.distributed or rank == 0:
        torch.cuda.synchronize()
        trainer.save(final_iter, checkpoint_path)
        print("==> Training complete. Final checkpoint saved at iter %d." % final_iter, flush=True)

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
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=200)
    parser.add_argument("--save_freq", type=int, default=5000,
                        help='检查点保存间隔 (步), 建议设为 2000-5000 以减少崩溃丢失')
    parser.add_argument("--wddm_sync_interval", type=int, default=100,
                        help='WDDM 防护: 每 N 步 synchronize+empty_cache (0=禁用, Windows 建议 100)')
    parser.add_argument("--exp_path", type=str, default='/data/ts/checkpoints/EDTalk/fine_tune/')
    parser.add_argument("--exp_name", type=str, default='Obama')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='DataLoader worker 进程数 (0=主线程加载)')
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
    parser.add_argument("--mask_reg_weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--mask_mouth_ratio", type=float, default=0.4, help=argparse.SUPPRESS)
    parser.add_argument("--beta1", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--d_weight_decay", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--r1_batch_size", type=int, default=4, help=argparse.SUPPRESS)
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
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
