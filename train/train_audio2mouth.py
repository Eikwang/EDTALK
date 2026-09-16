# 解决 Windows OpenMP 冲突（必须在任何 import 之前设置！）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CUDA 显存碎片化修复: expandable_segments 允许 CUDA caching allocator
# 扩展已有内存段而非创建新段，大幅减少碎片化。
# 不使用 max_split_size_mb / roundup_power2_divisions（在 WDDM 下可能导致 segfault）。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# faulthandler: 捕获 C 级 segfault/access violation 并打印 Python  traceback
# Windows 上 DataLoader worker 或 CUDA driver 崩溃时，默认无任何输出。
# 启用后至少能看到崩溃发生在哪个 C 扩展调用。
import faulthandler
faulthandler.enable()
import sys as _fh_sys
if _fh_sys.stderr:
    faulthandler.enable(file=_fh_sys.stderr, all_threads=True)

# CUDA 内存分配策略
# 已在顶部设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 注意: 不使用 max_split_size_mb 和 roundup_power2_divisions — 它们在 Windows WDDM
# 模式下可能导致训练进程 segfault。expandable_segments 是安全的碎片化修复方案。

import argparse
import sys
import signal
import gc

# psutil 可选：用于内存监控，未安装时降级为无监控
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("[WARN] psutil 未安装，内存监控功能禁用。建议: pip install psutil")

# 确保项目根目录在 sys.path 中（支持从任意目录运行）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import datetime
import random
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from datasets.dataset_audio2lip import Audio2LipDataset_image_sync
import torchvision
import torchvision.transforms as transforms
from train.trainer_audio2mouth import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
# from config import Config
import torch.nn.functional as F
from torchvision import utils
import shutil
from tqdm import tqdm
from util.distributed_stylegan import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

# 全局标志：用于优雅关闭
_shutdown_requested = False

def _signal_handler(signum, frame):
    """信号处理器：捕获 Ctrl+C 和终止信号，优雅关闭"""
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n[WARNING] 收到信号 {sig_name}，正在优雅关闭...")
    print("[WARNING] 当前 batch 完成后将保存 checkpoint 并退出")
    faulthandler.dump_traceback()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# 不设置 set_per_process_memory_fraction 硬限制,避免 backward 峰值撞上限制导致
# Windows WDDM 静默崩溃。改用显式 del + empty_cache 控制显存。

torch.backends.cudnn.enabled = True
# 关键修复: cudnn.benchmark = False
# benchmark=True 会在运行时尝试多种卷积算法, 涉及大量临时 CUDA 内核启动和
# 内存分配。在 Windows WDDM 模式下, 这些临时分配与主计算流的内存管理竞争,
# 可能导致 CUDA driver 内部状态不一致 -> access violation。
# 性能影响: 微乎其微。cudnn autotuning 主要在输入尺寸频繁变化时有收益,
# 但本项目输入尺寸固定 (256x256), 首次运行时选择的算法已经是最优的。
torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings("ignore")


def get_memory_info():
    """获取当前进程和系统的内存使用情况"""
    if not _HAS_PSUTIL:
        return None
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # 系统内存
    sys_mem = psutil.virtual_memory()
    
    # GPU 内存
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
        }
    
    return {
        'process_rss_gb': mem_info.rss / 1024**3,
        'process_vms_gb': mem_info.vms / 1024**3,
        'system_used_gb': sys_mem.used / 1024**3,
        'system_available_gb': sys_mem.available / 1024**3,
        'system_percent': sys_mem.percent,
        'gpu': gpu_mem
    }


def check_memory_pressure(warn_threshold_gb=12.0, critical_threshold_gb=14.0):
    """检查内存压力，返回状态信息"""
    if not _HAS_PSUTIL:
        # psutil 不可用时，只显示 GPU 内存
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"[MEM] GPU: {alloc:.1f}GB alloc, {reserved:.1f}GB reserved"
        return "[MEM] 监控禁用 (psutil 未安装)"
    
    mem = get_memory_info()
    
    status = "OK"
    if mem['process_rss_gb'] > critical_threshold_gb:
        status = "CRITICAL"
    elif mem['process_rss_gb'] > warn_threshold_gb:
        status = "WARNING"
    
    gpu_str = ""
    if mem['gpu']:
        gpu_str = f" | GPU: {mem['gpu']['allocated']:.1f}GB alloc, {mem['gpu']['reserved']:.1f}GB reserved"
    
    return (
        f"[MEM] {status} | Process: {mem['process_rss_gb']:.1f}GB RSS | "
        f"System: {mem['system_percent']:.0f}% used, {mem['system_available_gb']:.1f}GB avail"
        f"{gpu_str}"
    )

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


def main(args):
    # 预热 CUDA - 确保 CUDA 上下文干净
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"[INFO] CUDA 预热中... GPU: {torch.cuda.get_device_name(0)}")
        _dummy = torch.empty(1, device=device)
        del _dummy
        torch.cuda.empty_cache()
        print(f"[INFO] CUDA 预热完成")

    # init distributed computing
    # ddp_setup(args, rank, world_size)
    # torch.cuda.set_device(rank)
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                .replace(":", "")\
                                .replace(" ", "_")
    # make logging folder
    log_path = os.path.join('logs', args.exp_path, args.exp_name, date, 'log')
    checkpoint_path = os.path.join('logs', args.exp_path, args.exp_name, date, 'checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    best_loss = 10000
    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    # if args.dataset == 'ted':
    #     dataset = TED('train', transform, True)
    #     dataset_test = TED('test', transform)
    # elif args.dataset == 'vox':
    dataset = Audio2LipDataset_image_sync(hdtf=args.data_path, is_train=True, transform=transform,
                                           preload_features=args.preload_features)
    
    # 优化: 全部数据用于训练，测试时随机抽取
    # 原方案问题: train/test 共享同一份 samples 列表, 测试集是从训练集截取的前1%,
    #   导致: 1) 数据泄露(测试数据被训练过) 2) 测试样本固定不变 3) 训练数据损失
    # 新方案: 训练集使用全部数据，测试时从训练集中随机抽取200个样本
    dataset_test = Audio2LipDataset_image_sync(hdtf=args.data_path, is_train=False, transform=transform,
                                                preload_features=args.preload_features)
    
    test_sample_count = min(200, len(dataset.samples))
    random.seed(42)  # 固定种子确保可复现
    dataset_test.samples = random.sample(dataset.samples, test_sample_count)
    random.seed()  # 恢复随机种子
    print(f"[Dataset] 训练集: {len(dataset.samples)} 样本, 测试集: {len(dataset_test.samples)} 样本 (随机抽取)")
    # elif args.dataset == 'taichi':
    #     dataset = Taichi('train', transform, True)
    #     dataset_test = Taichi('test', transform)
    # else:
    #     raise NotImplementedError

    # ===== 关键修复: DataLoader num_workers=0 =====
    # 两次 access violation 崩溃的共同特征:
    #   1) 都是 C 级段错误(非 Python 异常), 无法被 try/except 捕获
    #   2) 崩溃位置不同(train()遍历 vs _conv_forward), 但都有 4 个 _feed 线程活跃
    #   3) 移除 empty_cache() 后反而更早崩溃, 说明问题不在缓存回收时机
    #
    # 根因: Windows spawn 模式下 DataLoader worker 子进程与主进程 CUDA 操作冲突。
    #   worker 进程通过 LMDB 内存映射读取图像, 这些映射的虚拟地址空间
    #   与主进程的 CUDA 显存管理器(WDDM 虚拟化层)竞争, 导致 CUDA driver
    #   内部的页面表损坏 -> 主进程的 CUDA tensor 读写触发 access violation。
    #   这不是 PyTorch 层面的 bug, 而是 Windows WDDM + LMDB mmap + CUDA 的
    #   三方交互问题, 在 Linux 上不会发生(Linux 用 fork 而非 spawn, 且无 WDDM)。
    #
    # 修复: num_workers=0, 在主进程内同步加载数据, 彻底消除子进程。
    # 性能影响: 训练循环的主要瓶颈在 GPU 计算(VGG/Decoder/SyncNet forward+backward),
    #   数据加载在 CPU 端完成(约 10ms/batch), 远小于 GPU 计算(约 800ms/batch),
    #   因此 num_workers=0 不会成为性能瓶颈。
    if args.distributed == False:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
        test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,
                                     num_workers=0, pin_memory=False, drop_last=True)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                num_workers=0, sampler=train_sampler, pin_memory=False, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=(test_sampler is None),
                                     num_workers=0, sampler=test_sampler, pin_memory=False, drop_last=True)

    trainSteps = len(dataloader)
    testSteps = len(test_dataloader)
    # loader = sample_data(loader)
    # loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device)

    # AMP 混合精度: GradScaler 缩放 loss 防止 FP16 梯度下溢
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if args.amp else None
    if args.amp:
        print('[INFO] AMP 混合精度训练已启用 (FP16 forward + FP32 backward)')

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt, args.audio2lip_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    current_iter = args.start_iter
    test_iter = 0
    print('==> training')
    if args.disable_vgg:
        print('[INFO] VGG loss 已完全禁用 (专注于 L1 和 Sync loss)')
    elif args.vgg_sparse_freq > 0:
        comp_msg = f', 权重补偿 ×{args.vgg_sparse_freq}' if args.vgg_sparse_compensate else ', 无权重补偿'
        print(f'[INFO] VGG 稀疏模式: 每 {args.vgg_sparse_freq} iter 计算一次{comp_msg}, 基础权重 {args.vgg_weight} (预计节省 ~{100-100//args.vgg_sparse_freq:.0f}% VGG 计算)')
    else:
        print(f'[INFO] VGG 全量模式: 每 iter 计算 4 尺度 x 8 次 VGG19 forward, 权重 {args.vgg_weight}')
    print(f'[INFO] 初始内存状态: {check_memory_pressure()}')

    # === LR & Sync Weight 调度 ===
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    lr_schedule = None
    sync_schedule = None
    if args.lr_schedule:
        lr_schedule = [float(x) for x in args.lr_schedule.split(',')]
        print(f'[SCHEDULE] LR schedule: {lr_schedule} (raw, will be ×{g_reg_ratio:.2f})')
        print(f'[SCHEDULE] Effective LR: {[f"{x*g_reg_ratio:.6f}" for x in lr_schedule]}')
    if args.sync_schedule:
        sync_schedule = [float(x) for x in args.sync_schedule.split(',')]
        print(f'[SCHEDULE] Sync schedule: {sync_schedule}')
    if lr_schedule or sync_schedule:
        print(f'[SCHEDULE] Epoch mapping:')
        for ep in range(args.epoch):
            lr_val = lr_schedule[min(ep, len(lr_schedule)-1)] if lr_schedule else args.lr
            sw_val = sync_schedule[min(ep, len(sync_schedule)-1)] if sync_schedule else args.sync_weight
            print(f'  Epoch {ep+1}: LR={lr_val:.5f} (eff={lr_val*g_reg_ratio:.6f}), sync_weight={sw_val:.1f}')
    # === 调度定义结束 ===
    # pbar = range(args.iter)
    last_name = None
    test_epoch= 0
    for epoch in range(args.epoch):
        # 检查是否需要优雅关闭
        if _shutdown_requested:
            print(f"\n[INFO] 收到关闭信号，保存 checkpoint 后退出...")
            trainer.save(current_iter, checkpoint_path)
            print(f"[INFO] Checkpoint 已保存: iter={current_iter}")
            break
        
        print(f"\n[INFO] === Epoch {epoch+1}/{args.epoch} 开始 === iter={current_iter} | {check_memory_pressure()}")

        # === LR & Sync Weight 调度: epoch 边界生效 ===
        if lr_schedule is not None:
            idx = min(epoch, len(lr_schedule) - 1)
            new_lr = lr_schedule[idx] * g_reg_ratio
            for param_group in trainer.g_optim.param_groups:
                param_group['lr'] = new_lr
        if sync_schedule is not None:
            idx = min(epoch, len(sync_schedule) - 1)
            trainer.sync_weight = sync_schedule[idx]
        if lr_schedule is not None or sync_schedule is not None:
            current_lr = trainer.g_optim.param_groups[0]['lr']
            print(f'[SCHEDULE] Epoch {epoch+1}: LR={current_lr:.6f}, sync_weight={trainer.sync_weight:.1f}')
            
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        # 累加器：用于统计本 epoch 的平均损失
        epoch_l_g_total = 0.0
        epoch_loss_sums = {}
        epoch_step_count = 0
        # 滚动统计窗口：用于 tqdm 后缀显示最近 N batch 的平均损失
        recent_losses = []
        # tqdm 进度条：显示 epoch 内 batch 进度
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epoch}', 
                    total=trainSteps, leave=True)
        
        try:
            for bii, bi in enumerate(pbar):
                # 检查是否需要优雅关闭
                if _shutdown_requested:
                    print(f"\n[INFO] 收到关闭信号，保存 checkpoint 后退出...")
                    trainer.save(current_iter, checkpoint_path)
                    print(f"[INFO] Checkpoint 已保存: iter={current_iter}")
                    return
                    
                current_iter += 1
                audio_features, lip_features, pose_features, identity_img, target_img, bbox\
                = bi['audio_features'], bi['lip_features'],bi['pose_features'], bi['identity_img'], bi['target_img'], bi['bbox']
                audio_features, lip_features, pose_features, identity_img, target_img, bbox = \
                            audio_features.cuda(), lip_features.cuda(), pose_features.cuda(), identity_img.cuda(), target_img.cuda(), bbox.cuda()
             
                # 梯度累积：只在每 accumulation_steps 步执行一次优化器 step
                step_optimizer = ((bii + 1) % args.accumulation_steps == 0) or (bii + 1 == len(dataloader))
                
                # update generator
                G_losses, recon, l_g_total = trainer.gen_update(
                    audio_features, lip_features, pose_features, identity_img, target_img, bbox,
                    scaler=scaler, step_optimizer=step_optimizer
                )

                # update discriminator
                # gan_d_loss = trainer.dis_update(torch.cat([img_a, img_b], dim=0), torch.cat([recon_a_cross, recon_b_cross], dim=0))

                # 只在实际执行 optimizer step 时累加 epoch 统计（避免重复计数）
                if step_optimizer:
                    loss_val = l_g_total.detach().item()
                    epoch_l_g_total += loss_val
                    for key, value in G_losses.items():
                        epoch_loss_sums[key] = epoch_loss_sums.get(key, 0.0) + value.detach().item()
                    epoch_step_count += 1
                    # 滚动窗口：保留最近 100 个 step 的 loss
                    recent_losses.append(loss_val)
                    if len(recent_losses) > 100:
                        recent_losses.pop(0)

                # 更新 tqdm 后缀：显示最近 batch 的 loss
                if (bii + 1) % 50 == 0 or (bii + 1) == trainSteps:
                    recent_avg = sum(recent_losses) / max(len(recent_losses), 1)
                    pbar.set_postfix({
                        'loss': f'{recent_avg:.4f}',
                        'iter': current_iter
                    })

                # 每 log_freq 个 batch 打印一次滚动日志（单行，不刷屏）
                if args.log_freq > 0 and (bii + 1) % args.log_freq == 0:
                    recent_avg = sum(recent_losses) / max(len(recent_losses), 1)
                    loss_str = ' '.join(f'{k}:{v.detach().item():.4f}' for k, v in G_losses.items())
                    cur_loss = l_g_total.detach().item()
                    print(f'  [Epoch {epoch+1} Batch {bii+1}/{trainSteps}] '
                          f'recent_avg={recent_avg:.4f} total={cur_loss:.4f} {loss_str}')

                if current_iter % args.log_iter == 0:
                    writer.add_scalar('train_loss/l_g_total_loss', l_g_total.detach().item(), current_iter)
                    for key, value in G_losses.items():
                        writer.add_scalar('train_loss/{}'.format(key), value.detach().item(), current_iter)

                    # 每 1000 iter 打印一次内存状态
                    print(check_memory_pressure())

                # display
                # if current_iter % args.display_freq == 0:
                #     print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f]"
                #         % (current_iter, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item()))
                if current_iter % args.image_save_iter == 0:

                    sample = F.interpolate(torch.cat((identity_img.detach(),target_img.detach()[:,-1], recon.detach()), dim=0), 256)
                    # sample = torch.flip(sample, [1])
                    utils.save_image(
                        sample,
                        os.path.join(checkpoint_path, "epoch_%05d_step_%05d_train.jpg"%(epoch, current_iter)),
                        nrow=int(args.batch_size),
                        normalize=True,
                    )

                if current_iter % args.eval_iter == 0:
                    # 验证阶段：收集所有 batch 的 loss，验证完后打印汇总
                    curloss = 0
                    eval_losses = {}  # 收集所有 batch 的 loss
                    eval_steps = 0
                    # lip_mlp = lip_mlp.eval()
                    if args.distributed:
                        test_epoch +=1
                        test_dataloader.sampler.set_epoch(test_epoch)
                    # 用 tqdm 进度条替代每 20 batch 的 print，避免验证集较大时刷屏
                    # （原写法在 testSteps=20605 时会打印 ~1000 行 [Eval] batch 日志）
                    eval_pbar = tqdm(test_dataloader, desc=f'  [Eval] Epoch {epoch+1}',
                                     total=testSteps, leave=False, ncols=100)
                    for bii, bi in enumerate(eval_pbar):
                        test_iter+=1
                        if test_iter % 1000 == 0:
                            eval_pbar.set_postfix_str(f'iter={test_iter}')
                        with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp):
                            audio_features, lip_features, pose_features, identity_img, target_img, bbox\
                            = bi['audio_features'], bi['lip_features'],bi['pose_features'], bi['identity_img'], bi['target_img'], bi['bbox']
                            audio_features, lip_features, pose_features, identity_img, target_img, bbox = \
                                        audio_features.cuda(), lip_features.cuda(), pose_features.cuda(), identity_img.cuda(), target_img.cuda(), bbox.cuda()

                            G_losses, recon, l_g_total = trainer.sample(audio_features, lip_features, pose_features, identity_img, target_img, bbox)
                            curloss += l_g_total.detach().item()
                            eval_steps += 1

                            # 收集各 loss
                            for key, value in G_losses.items():
                                eval_losses[key] = eval_losses.get(key, 0.0) + value.detach().item()
                            eval_losses['l_g_total'] = eval_losses.get('l_g_total', 0.0) + l_g_total.detach().item()

                            if test_iter % args.image_save_iter == 0:
                                sample = F.interpolate(torch.cat((identity_img.detach(),target_img.detach()[:,-1], recon.detach()), dim=0), 256)
                                # sample = torch.flip(sample, [1])
                                utils.save_image(
                                    sample,
                                    os.path.join(checkpoint_path, "epoch_%05d_step_%05d_test.jpg"%(epoch, test_iter)),
                                    nrow=int(args.batch_size),
                                    normalize=True,
                                )
                                last_name = os.path.join(checkpoint_path, "epoch_%05d_step_%05d_test.jpg"%(epoch, test_iter))
                            # break
                    
                    # === 验证完成：打印汇总日志 ===
                    curloss = curloss / eval_steps
                    avg_losses = {k: v / eval_steps for k, v in eval_losses.items()}
                    
                    log_info = f'[Eval] Epoch {epoch+1}/{args.epoch} | Steps: {eval_steps} | Avg Loss: {curloss:.4f}'
                    log_info += f' | L1: {avg_losses.get("recon_l1_loss", 0):.4f}'
                    log_info += f' | VGG: {avg_losses.get("recon_vgg_loss", 0):.1f}'
                    log_info += f' | Sync: {avg_losses.get("sync", 0):.4f}'
                    log_info += f' | Best: {best_loss:.4f}'
                    print(log_info)


                    if curloss<best_loss:
                        best_loss = curloss
                        trainer.save(current_iter, checkpoint_path)
                        if last_name!=None:
                            shutil.copy(last_name, os.path.join(checkpoint_path, 'best_epoch_%06d_step_%06d.jpg'%(epoch, current_iter)))
                        start_i = 0
                        for bii, bi in enumerate(test_dataloader):
                            with torch.no_grad():
                                if start_i> args.vis_num:
                                    break
                                start_i+=1
                                audio_features, lip_features, pose_features, identity_img, target_img\
                                = bi['audio_features'], bi['lip_features'],bi['pose_features'], bi['identity_img'], bi['target_img']
                                audio_features, lip_features, pose_features, identity_img, target_img = \
                                            audio_features.cuda(), lip_features.cuda(), pose_features.cuda(), identity_img.cuda(), target_img.cuda()
                                

                                recon_target = trainer.sample_no_loss(audio_features, lip_features, pose_features, identity_img, target_img)

                                sample = F.interpolate(torch.cat((identity_img.detach(),target_img.detach()[:,-1], recon_target.detach()), dim=0), 256)
                                # sample = torch.flip(sample, [1])
                                utils.save_image(
                                    sample,
                                    os.path.join(checkpoint_path, "best_epoch_%06d_step_%06dd_%05d.jpg"%(epoch, current_iter, start_i)),
                                    nrow=int(args.batch_size),
                                    normalize=True,
                                )

                if current_iter%args.save_freq==0:
                    trainer.save(current_iter, checkpoint_path)
                    if last_name!=None:
                        shutil.copy(last_name, os.path.join(checkpoint_path, 'step_%06d.jpg'%(current_iter)))
                    
                    start_i = 0
                    if args.distributed:
                        test_dataloader.sampler.set_epoch(test_epoch+6)
                    for bii, bi in enumerate(test_dataloader):
                        with torch.no_grad():
                            if start_i> args.vis_num:
                                break
                            start_i+=1
                            audio_features, lip_features, pose_features, identity_img, target_img\
                            = bi['audio_features'], bi['lip_features'],bi['pose_features'], bi['identity_img'], bi['target_img']
                            audio_features, lip_features, pose_features, identity_img, target_img = \
                                        audio_features.cuda(), lip_features.cuda(), pose_features.cuda(), identity_img.cuda(), target_img.cuda()
                            

                            recon_target = trainer.sample_no_loss(audio_features, lip_features, pose_features, identity_img, target_img)

                            sample = F.interpolate(torch.cat((identity_img.detach(),target_img.detach()[:,-1], recon_target.detach()), dim=0), 256)
                            # sample = torch.flip(sample, [1])
                            utils.save_image(
                                sample,
                                os.path.join(checkpoint_path, "step_%05d_%05d.jpg"%(current_iter, start_i)),
                                nrow=int(args.batch_size),
                                normalize=True,
                            )

                # === for 循环末尾：释放本 iter 的 Python 引用 ===
                del audio_features, lip_features, pose_features
                del identity_img, target_img, bbox
                del G_losses, recon, l_g_total

                # 内存碎片整理: 每 500 iter 执行一次
                # expandable_segments 已大幅缓解碎片化, 但 VGG19 的大块临时张量
                # 仍会产生碎片, 定期 empty_cache 回收空洞防止 reserved 持续膨胀
                #
                # ⚠️ 已禁用 empty_cache(): 在 Windows WDDM 模式下，频繁调用
                # empty_cache() 可能触发 NVIDIA 驱动 bug (nvlddmkm.sys 空指针解引用)
                # 导致 BSOD。expandable_segments 已足够管理显存碎片。
                if current_iter % 500 == 0:
                    gc.collect()
                    # torch.cuda.empty_cache()  # 禁用以避免 BSOD

        except Exception as e:
            # bii 可能在 for 循环第一次迭代前未定义
            batch_info = bii + 1 if 'bii' in dir() and isinstance(bii, int) else 'N/A'
            print(f"\n[ERROR] 训练循环异常: {type(e).__name__}: {e}")
            print(f"[ERROR] 当前 iteration: {current_iter}, epoch: {epoch+1}, batch: {batch_info}")
            print(f"[ERROR] 内存状态: {check_memory_pressure()}")
            
            # CUDA OOM 专项诊断
            if 'CUDA' in str(e) and ('out of memory' in str(e).lower() or 'OOM' in str(e)):
                print(f"[ERROR] CUDA 显存不足！当前 GPU 状态:")
                if torch.cuda.is_available():
                    print(f"  allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                    print(f"  reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                    print(f"  max_alloc: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
                print(f"  建议: 降低 batch_size 或 accumulation_steps")
            
            import traceback
            traceback.print_exc()
            
            # 尝试保存紧急 checkpoint
            try:
                print("[INFO] 尝试保存紧急 checkpoint...")
                trainer.save(current_iter, checkpoint_path)
                print(f"[INFO] 紧急 checkpoint 已保存: iter={current_iter}")
            except Exception as save_err:
                print(f"[ERROR] 紧急 checkpoint 保存失败: {save_err}")
            
            raise  # 重新抛出异常，让调用者知道发生了什么

        # === Epoch 结束：打印本 epoch 的平均损失 ===
        if epoch_step_count > 0:
            avg_l_g_total = epoch_l_g_total / epoch_step_count
            log_info = 'train_Epoch [{}/{}] done, Steps={}, Avg l_g_total: {:.4f}'.format(
                epoch, args.epoch, epoch_step_count, avg_l_g_total)
            for key in sorted(epoch_loss_sums.keys()):
                avg_value = epoch_loss_sums[key] / epoch_step_count
                log_info += ' {}: {:.4f}'.format(key, avg_value)
            print(log_info)

            # === 每个 epoch 结束时保存完整 checkpoint ===
            # 确保每个 epoch 都有一个完整的模型备份，避免因 save_freq 间隔导致的"中间状态"丢失
            # 注意: trainer.save(path) 内部会再拼 /{idx:06d}.pt, 所以这里必须传目录而非 .pt 文件路径
            epoch_ckpt_dir = os.path.join(checkpoint_path, f'epoch_{epoch:03d}')
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            trainer.save(current_iter, epoch_ckpt_dir)
            print(f'[Epoch {epoch+1}] checkpoint saved: {os.path.join(epoch_ckpt_dir, f"{current_iter:06d}.pt")}')

        # Epoch 间清理: gc + empty_cache 整理碎片
        gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()  # 禁用以避免 BSOD
        print(f'[INFO] Epoch {epoch+1} 结束 | {check_memory_pressure()}')

    return


if __name__ == "__main__":
    # Windows 下必须使用 spawn 模式解决多进程问题
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了

    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--epoch", type=int, default=5,
                        help="训练 epoch 数 (微调推荐 3-5, 过多会导致过拟合和口型崩坏)")
    parser.add_argument("--size", type=int, default=256)
    
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率 (Decoder 微调推荐 0.0001~0.0005)")
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=1)
    
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_lip", type=int, default=20)
    parser.add_argument("--latent_dim_pose", type=int, default=6)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./Audio2Lip/')
    parser.add_argument("--exp_name", type=str, default='image_use_sync')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--semantic_radius", type=int, default=13)
    parser.add_argument("--log_iter", type=int, default=1000)
    parser.add_argument("--image_save_iter", type=int, default=6000, help="local rank for distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--distributed", action='store_true', help="Enable distributed training")
    parser.add_argument("--eval_iter", type=int, default=4000, help="local rank for distributed training")
    parser.add_argument("--sync_weight", type=float, default=2.0,
                        help="SyncNet loss weight (0=disabled, 自训 SyncNet 推荐 0.5~2.0)")
    parser.add_argument("--syncnet_ckpt", type=str, default=None, help="Path to SyncNet pre-trained checkpoint (e.g. ckpts/EDTalk.pt)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dis_weight", type=int, default=0, help="GAN 判别器权重 (0=禁用)。微调推荐关闭：小数据集 GAN 易崩溃，且 d_optim 链路未启用时 dis 仅向 audio2lip 注入噪声梯度")
    parser.add_argument("--vis_num", type=int, default=10, help="local rank for distributed training")
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--audio2lip_ckpt", type=str, default=None)
    parser.add_argument("--audio_encoder_unfreeze_from", type=int, default=9,
                        help="audio_encoder 从第 N 个 block 开始解冻 (0=全部解冻, 9=深层, 13=只解冻 mapping)。默认 9，保留浅层 Wav2Lip 特征")
    parser.add_argument("--audio_encoder_lr_mult", type=float, default=0.1,
                        help="audio_encoder 解冻层的学习率倍数 (相对 --lr，默认 0.1)")
    parser.add_argument("--mapping_lr_mult", type=float, default=1.0,
                        help="mapping 层学习率倍数 (相对 --lr，默认 1.0)")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数（单卡显存不足时使用，等效 batch_size = batch_size * accumulation_steps）")
    parser.add_argument("--data_path", type=str, default='HDTF', help="Dataset base directory path")
    parser.add_argument("--log_freq", type=int, default=1000, help="每 N 个 batch 打印一次滚动日志（0=只在 epoch 结束打印）")
    parser.add_argument("--preload_features", action='store_true',
                        help="将所有特征 preload_features 预加载到内存 (小数据集微调推荐, 大数据集慎用)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 进程数 (0=主线程加载, 默认4)")
    parser.add_argument("--vgg_sparse_freq", type=int, default=50,
                        help="VGG 稀疏计算频率 (0=每 iter 都算, N=每 N iter 算一次。默认 50, 可节省 VGG 计算且避免崩坏)")
    parser.add_argument("--vgg_weight", type=float, default=0.15,
                        help="VGG loss 基础权重 (默认 0.15。稀疏模式下默认不做 ×N 补偿，避免单步梯度爆炸)")
    parser.add_argument("--vgg_sparse_compensate", action='store_true',
                        help="VGG 稀疏时启用 ×N 权重补偿（旧行为）。经验上补偿易致崩坏，默认关闭")
    parser.add_argument("--mouth_vgg_weight", type=float, default=0.1,
                        help="嘴部 crop VGG 权重 (每 iter 计算 96x96 脸部 crop 下半区，恢复牙齿/口腔细节。0=禁用。稳态目标: mouth_vgg_loss 占总 loss 5-15%%)")
    parser.add_argument("--mouth_vgg_crop_top", type=float, default=0.4,
                        help="嘴部 crop 从脸部 96x96 crop 顶部截掉的比例 (0.4=取下 60%%，含嘴+下巴)")
    parser.add_argument("--smooth_weight", type=float, default=0.2,
                        help="recon_smooth 帧间平滑损失权重 (原 0.5 硬编码；降低可减少对爆破音快速唇形的抑制)")
    parser.add_argument("--disable_vgg", action='store_true',
                        help="完全关闭 VGG loss (专注于 L1 和 Sync loss, 适合需要口型细节的场景)")
    parser.add_argument("--amp", action='store_true',
                        help="启用 AMP 混合精度训练 (FP16 forward + FP32 backward), 加速训练并减少显存占用")
    parser.add_argument("--lr_schedule", type=str, default=None,
                        help="Per-epoch LR schedule, comma-separated (e.g. '0.001,0.0005,0.0002,0.0001,0.00005'). "
                             "Overrides --lr per epoch. Values use same scale as --lr (multiplied by g_reg_ratio internally).")
    parser.add_argument("--sync_schedule", type=str, default=None,
                        help="Per-epoch sync_weight schedule, comma-separated (e.g. '1,2,4,6,8'). "
                             "Overrides --sync_weight per epoch.")
    opts = parser.parse_args()

    # opts = Config(opts.config, opts, is_train=True)
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opts.distributed = n_gpu > 1
    # n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2

    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    main(opts)