import argparse
import os
import copy
import sys
import faulthandler

faulthandler.enable(file=sys.stderr, all_threads=True)

# CUDA 内存分配策略:必须在 import torch 之前设置
# expandable_segments 减少内存碎片,但 Windows 平台不支持 (仅 Linux),
# 强行设置只会产生警告。Windows 改用 garbage_collection_threshold 降低碎片。
import platform
if platform.system() == 'Windows':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6'
else:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

# 不设置 set_per_process_memory_fraction 硬限制。
# 实测 batch_size=8 时 backward 峰值 14.19 GB,0.90 限制仅允许 14.4 GB,
# 仅剩 210 MB 余量。display/val/save 操作叠加时极易撞上硬限制,
# 在 Windows WDDM 模式下触发静默 access violation (无 Python 异常)。
# 改用显式 del + empty_cache 控制显存,给 PyTorch 完整 16 GB 可用空间。

torch.backends.cudnn.enabled = True
# benchmark=False: 禁用 cuDNN 自动算法选择。benchmark 模式会在首次运行时
# 测试多种 cuDNN 算法并缓存最快的,但某些算法在 Windows + RTX 40 系列上
# 存在 bug,长时间运行后会触发 access violation。deterministic 模式使用
# 固定算法,牺牲少量性能换取稳定性。
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data

    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)


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
        print(f'[ERROR] count/ subdirectory not found in dataset: {args.datapath}')
        raise FileNotFoundError(f'count/ subdirectory not found in: {args.datapath}')
    print(f'  Using count directory: {count_dir}')

    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    dataset = Finetune256(count_dir, True, transform=transform, temporal_window=args.temporal_window)
    dataset_test = Finetune256(count_dir, False, transform=transform, temporal_window=args.temporal_window)

    if args.distributed:
        loader = data.DataLoader(
            dataset,
            num_workers=2,
            batch_size=args.batch_size // world_size,
            sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        loader_test = data.DataLoader(
            dataset_test,
            num_workers=2,
            batch_size=4,
            sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
    else:
        # NOTE: train loader 关闭 persistent_workers + num_workers=0，避免 Windows 上
        # persistent_workers + CUDA 初始化导致的死锁。
        loader = data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )
        # NOTE: test loader 关闭 persistent_workers，避免 Windows 上
        # persistent_workers + eval 迭代导致的死锁 (与 audio2mouth 流程同类问题)。
        loader_test = data.DataLoader(
            dataset_test,
            num_workers=0,
            batch_size=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
        )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # resume: 先从 checkpoint 文件名解析 start_iter, 让 Trainer 的 scheduler 正确初始化
    if args.resume_ckpt is not None:
        ckpt_name = os.path.basename(args.resume_ckpt)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            args.start_iter = 0
    else:
        args.start_iter = 0

    trainer = Trainer(args, device, rank)

    # 加载模型权重和 optimizer 状态
    if args.resume_ckpt is not None:
        start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (start_iter))

    print('==> training')
    pbar = range(args.iter)

    # 验证集监控 + best 模型 + 早停 状态
    best_val_loss = float('inf')
    best_gen_state = None
    patience_counter = 0
    val_history = []  # 记录 (iter, val_loss) 供观察

    for idx in pbar:
        i = idx + args.start_iter

        try:
            # loading data
            img_source, img_target = next(loader)
            img_source = img_source.to(device, non_blocking=True)
            img_target = img_target.to(device, non_blocking=True)

            # update generator
            vgg_loss, l1_loss, gan_g_loss, id_loss, img_recon = trainer.gen_update(img_source, img_target)

            # update discriminator (含 R1 lazy regularization)
            # R1 使用小 batch + 分离 backward 降低 Windows 崩溃风险
            do_r1 = (args.d_reg_every > 0 and i % args.d_reg_every == 0)
            gan_d_loss, r1_penalty = trainer.dis_update(img_target, img_recon, do_r1=do_r1)

            # 释放 img_recon 的计算图 (含完整 gen forward 中间激活)。
            # 必须在 display/val 之前执行,否则训练图 + 测试前向同时驻留显存,
            # batch_size=8 时峰值超 14 GB 导致 Windows WDDM 静默崩溃。
            img_recon = img_recon.detach()

            # LR 调度器步进
            trainer.g_scheduler.step()
            trainer.d_scheduler.step()

        except Exception as e:
            print(f"\n[ERROR] Training crashed at iteration {i}!", file=sys.stderr, flush=True)
            print(f"[ERROR] Exception: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"[ERROR] Saving emergency checkpoint...", file=sys.stderr, flush=True)
            try:
                trainer.save(f'emergency_{i}', checkpoint_path)
                print(f"[ERROR] Emergency checkpoint saved to {checkpoint_path}/emergency_{i}.pt", file=sys.stderr, flush=True)
            except:
                pass
            raise

        # 定期清理 CUDA 缓存,减少 Windows 上的内存碎片导致的 access violation
        if i % 100 == 0 and i > 0:
            torch.cuda.empty_cache()

        # 每 500 次迭代打印进度 (display_freq 默认 2500,中间无输出,崩溃时无法定位)
        if i % 500 == 0 and i > 0 and (not args.distributed or rank == 0):
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print("[Progress Iter %d/%d] [vgg: %.2f] [l1: %.4f] [g: %.4f] [d: %.4f] [mem: alloc=%.2fGB reserved=%.2fGB]"
                  % (i, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item(),
                     mem_alloc, mem_reserved), flush=True)

        # write to log (single GPU always logs; multi GPU only rank 0)
        if not args.distributed or rank == 0:
            write_loss(idx, vgg_loss, l1_loss, gan_g_loss, gan_d_loss, writer)
            writer.add_scalar('r1_penalty', r1_penalty.item(), idx)
            writer.add_scalar('lr', trainer.g_optim.param_groups[0]['lr'], idx)
            if isinstance(id_loss, torch.Tensor):
                writer.add_scalar('id_loss', id_loss.item(), idx)

        # display
        if i % args.display_freq == 0 and (not args.distributed or rank == 0):
            with torch.no_grad():
                id_loss_val = id_loss.item() if isinstance(id_loss, torch.Tensor) else 0.0
                print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f] [id loss: %f] [lr: %.2e]"
                      % (i, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item(), id_loss_val,
                         trainer.g_optim.param_groups[0]['lr']))

                img_test_source, img_test_target = next(loader_test)
                img_test_source = img_test_source.to(device, non_blocking=True)
                img_test_target = img_test_target.to(device, non_blocking=True)

                img_recon_test = trainer.sample(img_test_source, img_test_target)

                sample = F.interpolate(torch.cat((img_test_source.detach(), img_test_target.detach(), img_recon_test.detach()), dim=0), 256)
                utils.save_image(
                    sample,
                    os.path.join(checkpoint_path, "step_%05d.jpg" % (i)),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                del img_test_source, img_test_target, img_recon_test, sample

        # 验证集监控 + best 模型保存 + 早停
        if i % args.val_freq == 0 and i > 0 and (not args.distributed or rank == 0):
            with torch.no_grad():
                val_vgg_sum, val_l1_sum, val_id_sum, val_n = 0.0, 0.0, 0.0, 0
                # 遍历验证集前若干个 batch 计算 平均指标 (限制 batch 数避免过久)
                max_val_batches = 10
                for v_idx, (v_src, v_tgt) in enumerate(loader_test):
                    if v_idx >= max_val_batches:
                        break
                    v_src = v_src.to(device, non_blocking=True)
                    v_tgt = v_tgt.to(device, non_blocking=True)
                    v_vgg, v_l1, v_id = trainer.validate(v_src, v_tgt)
                    val_vgg_sum += v_vgg.item()
                    val_l1_sum += v_l1.item()
                    val_id_sum += v_id.item() if isinstance(v_id, torch.Tensor) else 0.0
                    val_n += 1
                    del v_src, v_tgt, v_vgg, v_l1, v_id

            if val_n > 0:
                val_vgg = val_vgg_sum / val_n
                val_l1 = val_l1_sum / val_n
                val_id = val_id_sum / val_n
                val_loss = val_vgg + val_l1  # 验证指标主项
                val_history.append((i, val_loss))
                writer.add_scalar('val/vgg_loss', val_vgg, i)
                writer.add_scalar('val/l1_loss', val_l1, i)
                writer.add_scalar('val/id_loss', val_id, i)
                writer.add_scalar('val/total_loss', val_loss, i)

                print("[VAL Iter %d] [vgg: %f] [l1: %f] [id: %f] [best: %f]"
                      % (i, val_vgg, val_l1, val_id, best_val_loss))

                # best 模型判定 (基于验证集 total loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_gen_state = copy.deepcopy(trainer.gen.state_dict())
                    patience_counter = 0
                    # 实时保存 best 模型
                    trainer.save('best', checkpoint_path)
                    print("  >> New best model saved (val_loss=%.4f)" % val_loss)
                else:
                    patience_counter += 1
                    print("  >> No improvement (%d/%d)" % (patience_counter, args.patience))

                # 早停
                if args.patience > 0 and patience_counter >= args.patience:
                    print("[EARLY STOP] No validation improvement for %d checks. Stopping." % args.patience)
                    break

        # save model
        if i % args.save_freq == 0 and (not args.distributed or rank == 0):
            trainer.save(i, checkpoint_path)

        # 每轮迭代末尾显式清理,防止上一轮的中间张量残留到下一轮
        del img_source, img_target, img_recon
        del vgg_loss, l1_loss, gan_g_loss, gan_d_loss, id_loss
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # 训练结束: 用 best 模型覆盖最终保存 (若有)
    if best_gen_state is not None and (not args.distributed or rank == 0):
        trainer.gen.load_state_dict(best_gen_state)
        trainer.save('final_best', checkpoint_path)
        print("==> Training done. Best model (val_loss=%.4f) saved as final_best." % best_val_loss)

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only_fine_tune_dec", action='store_true', help='Only fine tune dec in Generator')
    parser.add_argument("--d_reg_every", type=int, default=16,
                        help='R1 正则间隔 (0=禁用, 默认 16 lazy regularization)')
    parser.add_argument("--r1_batch_size", type=int, default=2,
                        help='R1 正则计算使用的 batch 大小 (降低二阶梯度内存, 建议 1-4)')
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default='ckpts/EDTalk_lip_pose.pt')
    parser.add_argument("--datapath", type=str, required=True,
                        help='Dataset directory (must contain count/ subdirectory with frame images)')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup_iters", type=int, default=500, help='LR warmup 步数')
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=500, help='控制台输出和预览图生成间隔 (步)')
    parser.add_argument("--save_freq", type=int, default=1000, help='检查点保存间隔 (步)')
    parser.add_argument("--val_freq", type=int, default=2000, help='验证集监控间隔 (步)')
    parser.add_argument("--patience", type=int, default=15, help='早停耐心值 (验证集无改善次数，0=禁用早停)')
    parser.add_argument("--temporal_window", type=int, default=30, help='时序邻近采样窗口 (source±window)，越大越随机')
    parser.add_argument("--exp_path", type=str, default='/data/ts/checkpoints/EDTalk/fine_tune/')
    parser.add_argument("--exp_name", type=str, default='Obama')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--id_weight", type=float, default=0.5, help='身份保持损失权重（0=禁用），latent 空间约束身份')
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
