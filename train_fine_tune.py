import argparse
import os
import random
import sys
import faulthandler
import multiprocessing
import platform

# Windows 多进程: 必须使用 spawn 方式 (fork 在 Windows 上不可用)
# 必须在 import torch 之前设置，否则 DataLoader 子进程可能继承错误的 CUDA 状态
if platform.system() == 'Windows':
    multiprocessing.set_start_method('spawn', force=True)

# 抑制 TensorFlow 警告 (VGG19 Loss + TensorBoard 间接触发)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

faulthandler.enable(file=sys.stderr, all_threads=True)

if platform.system() == 'Windows':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6'
else:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch.utils import data
from fine_tune.dataset import Finetune256
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from fine_tune.trainer_fine_tune import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision import utils

# 不设置 set_per_process_memory_fraction 硬限制。
# 改用显式 del 控制显存,给 PyTorch 完整可用空间。

torch.backends.cudnn.enabled = True
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

    dataset = Finetune256(count_dir, True, transform=transform, temporal_window=args.temporal_window,
                          cross_window_min=args.cross_window_min, cross_window_max=args.cross_window_max,
                          far_window=args.far_window, far_ratio=args.far_ratio)
    dataset_test = Finetune256(count_dir, False, transform=transform, temporal_window=args.temporal_window,
                               cross_window_min=args.cross_window_min, cross_window_max=args.cross_window_max,
                               far_window=args.far_window, far_ratio=0.0)

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
        num_workers = getattr(args, 'num_workers', 0)
        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )
        if num_workers > 0:
            loader_kwargs['num_workers'] = num_workers
            loader_kwargs['prefetch_factor'] = 2
        loader = data.DataLoader(dataset, **loader_kwargs)
        loader_test_kwargs = dict(
            batch_size=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
        )
        if num_workers > 0:
            loader_test_kwargs['num_workers'] = num_workers
            loader_test_kwargs['prefetch_factor'] = 2
        loader_test = data.DataLoader(dataset_test, **loader_test_kwargs)

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

    # 加载模型权重和 optimizer 状态 (正交正则不改变参数结构, ckpt 完全兼容)
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

        # BUG-5 修复: DataLoader worker 崩溃时重建并重试, 防止训练直接退出
        try:
            img_data = next(loader)
        except (StopIteration, RuntimeError) as e:
            print(f"\n[WARNING] DataLoader failed at iteration {i}: {e}", file=sys.stderr, flush=True)
            print("[WARNING] Attempting to recreate DataLoader...", file=sys.stderr, flush=True)
            if args.distributed:
                loader = data.DataLoader(
                    dataset, num_workers=2,
                    batch_size=args.batch_size // world_size,
                    sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
                    pin_memory=True, drop_last=False, persistent_workers=True,
        
                )
            else:
                recovery_kwargs = dict(
                    batch_size=args.batch_size, shuffle=True,
                    pin_memory=False, drop_last=False, persistent_workers=False,
                )
                nw = getattr(args, 'num_workers', 0)
                if nw > 0:
                    recovery_kwargs['num_workers'] = nw
                    recovery_kwargs['prefetch_factor'] = 2
                loader = data.DataLoader(dataset, **recovery_kwargs)
            loader = sample_data(loader)
            try:
                img_data = next(loader)
            except Exception:
                print("[ERROR] DataLoader recovery failed. Saving checkpoint and exiting.", file=sys.stderr, flush=True)
                try:
                    trainer.save(f'emergency_{i}', checkpoint_path)
                except:
                    pass
                raise
            print("[WARNING] DataLoader recovered, continuing training.", file=sys.stderr, flush=True)

        try:
            img_source, img_target = img_data[0], img_data[1]
            img_cross = img_data[2] if len(img_data) > 2 else None
            img_source = img_source.to(device, non_blocking=True)
            img_target = img_target.to(device, non_blocking=True)
            img_cross = img_cross.to(device, non_blocking=True) if img_cross is not None else None

            # update generator
            result = trainer.gen_update(img_source, img_target, img_cross)
            vgg_loss, l1_loss, gan_g_loss, id_loss, img_recon = result[0], result[1], result[2], result[3], result[4]
            cross_vgg, cross_l1, lip_space, pose_space, mask_reg = result[5], result[6], result[7], result[8], result[9]

            # update discriminator (weight_decay 一阶正则替代 R1, 无二阶梯度)
            gan_d_loss = trainer.dis_update(img_target, img_recon)

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
            writer.add_scalar('lr', trainer.g_optim.param_groups[0]['lr'], idx)
            if isinstance(id_loss, torch.Tensor):
                writer.add_scalar('id_loss', id_loss.item(), idx)
            if isinstance(cross_vgg, torch.Tensor) and cross_vgg.item() > 0:
                writer.add_scalar('cross_vgg', cross_vgg.item(), idx)
                writer.add_scalar('cross_l1', cross_l1.item(), idx)
            if isinstance(lip_space, torch.Tensor) and lip_space.item() > 0:
                writer.add_scalar('lip_space', lip_space.item(), idx)
                writer.add_scalar('pose_space', pose_space.item(), idx)

        # display
        if i % args.display_freq == 0 and (not args.distributed or rank == 0):
            with torch.no_grad():
                id_loss_val = id_loss.item() if isinstance(id_loss, torch.Tensor) else 0.0
                print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f] [id loss: %f] [cross: %f] [lr: %.2e]"
                      % (i, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item(), id_loss_val,
                         cross_vgg.item() if isinstance(cross_vgg, torch.Tensor) else 0.0,
                         trainer.g_optim.param_groups[0]['lr']))

                test_data = next(loader_test)
                img_test_source = test_data[0].to(device, non_blocking=True)
                img_test_target_near = test_data[1].to(device, non_blocking=True)
                img_test_cross = test_data[2].to(device, non_blocking=True)

                # 方案 B: 从不同视频组取 target，构造跨视频重建对，增大嘴型差异
                # dataset_test.group_ranges 记录每个视频组在 frames_paths 中的 [start, end)
                # 注意: dataset_test.__getitem__ 忽略传入的 idx, 内部随机采样 source,
                #       因此不能通过 dataset_test[idx] 按索引获取帧。
                #       改为直接从 frames_paths 按路径加载, 手动应用 transform。
                # 需要构造 batch_size 个跨视频样本, 与 loader_test 的 batch 维度对齐。
                n_groups = len(dataset_test.group_ranges)
                display_batch = img_test_source.shape[0]
                if n_groups >= 2:
                    src_list, tgt_list = [], []
                    for _ in range(display_batch):
                        src_global = random.randint(0, len(dataset_test.frames_paths) - 1)
                        src_group = dataset_test.frame_to_group[src_global]
                        other_groups = [g for g in range(n_groups) if g != src_group]
                        tgt_group = random.choice(other_groups)
                        tgt_start, tgt_end = dataset_test.group_ranges[tgt_group]
                        tgt_global = random.randint(tgt_start, tgt_end - 1)

                        src_img = Image.open(dataset_test.frames_paths[src_global]).convert('RGB')
                        tgt_img = Image.open(dataset_test.frames_paths[tgt_global]).convert('RGB')
                        src_list.append(transform(src_img))
                        tgt_list.append(transform(tgt_img))

                    img_test_source_cross_video = torch.stack(src_list).to(device, non_blocking=True)
                    img_test_target_cross_video = torch.stack(tgt_list).to(device, non_blocking=True)
                else:
                    # 只有一个视频组，退化为使用 test_data 的样本
                    img_test_source_cross_video = img_test_source
                    img_test_target_cross_video = img_test_target_near

                img_recon_test = trainer.sample(img_test_source, img_test_target_near)

                # 方案 D: 交叉重建 — source + cross_target(同视频远距离帧) → 应保持 source 背景
                # 检查 cross_target 是否有效 (非 target 克隆)
                has_valid_cross = not torch.equal(img_test_cross, img_test_target_near)
                if has_valid_cross:
                    img_cross_recon_test = trainer.sample(img_test_source, img_test_cross)
                else:
                    img_cross_recon_test = img_test_target_near  # 无有效 cross，使用 target 占位

                # 方案 B: 跨视频重建 — source(组A) + target(组B) → 测试解耦泛化能力
                img_cross_video_recon = trainer.sample(
                    img_test_source_cross_video, img_test_target_cross_video
                )

                # 预测图: 4 列 x 5 行
                #   row 1 = source              (身份+背景参考帧)
                #   row 2 = target (同视频近邻)  (期望重建目标，嘴型差异小)
                #   row 3 = recon (同视频近邻)    (正常重建: 验证嘴型准确度)
                #   row 4 = cross_recon           (交叉重建: source+cross→source，验证背景保持)
                #   row 5 = cross_video_recon     (跨视频重建: source(A)+target(B)，验证解耦泛化)
                #
                # 观察要点:
                #   - row1 vs row4: 背景/衣服应一致 (交叉重建 → 背景解耦)
                #   - row1 vs row5: 背景/衣服应一致 (跨视频重建 → 泛化解耦)
                #   - row2 vs row3: 嘴型应一致 (正常重建准确度)
                #   - row5 嘴型应接近 row5 target 的嘴型 (跨视频 direction 迁移)
                sample = F.interpolate(torch.cat((
                    img_test_source.detach(),
                    img_test_target_near.detach(),
                    img_recon_test.detach(),
                    img_cross_recon_test.detach(),
                    img_cross_video_recon.detach(),
                ), dim=0), 256)
                utils.save_image(
                    sample,
                    os.path.join(checkpoint_path, "step_%05d.jpg" % (i)),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                del img_test_source, img_test_target_near, img_recon_test, img_test_cross, img_cross_recon_test
                del img_test_source_cross_video, img_test_target_cross_video, img_cross_video_recon, sample

        # 验证集监控 + best 模型保存 + 早停
        if i % args.val_freq == 0 and i > 0 and (not args.distributed or rank == 0):
            with torch.no_grad():
                val_vgg_sum, val_l1_sum, val_id_sum, val_n = 0.0, 0.0, 0.0, 0
                # 遍历验证集前若干个 batch 计算 平均指标 (限制 batch 数避免过久)
                max_val_batches = 10
                for v_idx, v_data in enumerate(loader_test):
                    if v_idx >= max_val_batches:
                        break
                    v_src, v_tgt = v_data[0], v_data[1]  # 忽略 v_data[2] (cross) 在验证中
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
                    best_gen_state = {k: v.cpu() for k, v in trainer.gen.state_dict().items()}
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
        del cross_vgg, cross_l1, lip_space, pose_space, mask_reg

    # 训练结束: 用 best 模型覆盖最终保存 (若有)
    if best_gen_state is not None and (not args.distributed or rank == 0):
        trainer.gen.load_state_dict({k: v.to(device) for k, v in best_gen_state.items()})
        trainer.save('final_best', checkpoint_path)
        print("==> Training done. Best model (val_loss=%.4f) saved as final_best." % best_val_loss)

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=6,
                        help='batch size (WDDM 16GB 显存建议 4-6, 避免接近显存上限触发静默损坏)')
    parser.add_argument("--only_fine_tune_dec", action='store_true',
                        help='仅微调 Decoder (少帧模式: 冻结 Encoder+Direction+fc, 仅训练 Decoder)')
    parser.add_argument("--freeze_direction", action='store_true',
                        help='冻结 Direction+fc+lip_fc+pose_fc, 训练 Encoder+Decoder (多帧模式增强, 仅在全量微调时生效)')
    parser.add_argument("--d_reg_every", type=int, default=0,
                        help='[已废弃] R1 二阶梯度在 WDDM 上累积 native 内存损坏导致 access violation, 已改用 weight_decay; 保留仅为向后兼容')
    parser.add_argument("--r1_batch_size", type=int, default=4,
                        help='[已废弃] 同 d_reg_every, 保留仅为向后兼容')
    parser.add_argument("--d_weight_decay", type=float, default=0.01,
                        help='判别器 Adam weight_decay (替代 R1, 一阶 L2 正则, 抑制判别器过强防生成器被逼模糊, 0=禁用, 默认0.01)')
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
    parser.add_argument("--patience", type=int, default=20, help='早停耐心值 (验证集无改善次数，0=禁用早停)')
    parser.add_argument("--temporal_window", type=int, default=30, help='时序邻近采样窗口 (source±window)，越大越随机')
    parser.add_argument("--far_window", type=int, default=500,
                        help='远距离采样窗口 (source±far_window，排除temporal_window内帧)，扩大direction泛化范围')
    parser.add_argument("--far_ratio", type=float, default=0.7,
                        help='远距离采样概率 (0=全部邻近采样，1=全部远距离采样，默认0.3=30%%远距离+70%%邻近)')
    parser.add_argument("--exp_path", type=str, default='/data/ts/checkpoints/EDTalk/fine_tune/')
    parser.add_argument("--exp_name", type=str, default='Obama')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--id_weight", type=float, default=0.5, help='身份保持损失权重（0=禁用），latent 空间约束身份')
    parser.add_argument("--beta1", type=float, default=None,
                        help='Adam beta1 (动量系数，0=无动量/原训练风格，0.5=默认。留空则自动推断: only_fine_tune_dec->0, 全量->0.5)')
    parser.add_argument("--cross_weight", type=float, default=0.5,
                        help='交叉重建损失权重 (方案A，0=禁用，默认0.3)')
    parser.add_argument("--lip_space_weight", type=float, default=0.1,
                        help='lip空间解耦损失权重 (方案A，0=禁用，默认0.1)')
    parser.add_argument("--pose_space_weight", type=float, default=0.5,
                        help='pose空间解耦损失权重 (方案A，0=禁用，默认0.1)')
    parser.add_argument("--cross_window_min", type=int, default=None,
                        help='cross_target 最小窗口 (默认 temporal_window*2)')
    parser.add_argument("--cross_window_max", type=int, default=None,
                        help='cross_target 最大窗口 (默认 temporal_window*4)')
    parser.add_argument("--mask_reg_weight", type=float, default=0.0,
                        help='Mask 正则化权重 (方案B，0=禁用，建议0.05-0.1)')
    parser.add_argument("--mask_mouth_ratio", type=float, default=0.4,
                        help='嘴部区域占图像高度的底部比例 (mask 正则化豁免区域，默认0.4)')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='DataLoader worker 进程数 (0=主线程加载, 默认2)')
    opts = parser.parse_args()

    # 自动推断逻辑
    #   only_fine_tune_dec (少帧): 冻结 Encoder+Direction+fc, 仅训练 Decoder
    #   全量微调 + freeze_direction (多帧): 冻结 Direction+fc, 训练 Encoder+Decoder
    #
    #   freeze_direction 仅在全量微调时生效，与 only_fine_tune_dec 互不干扰
    if opts.only_fine_tune_dec:
        # 少帧模式: freeze_direction 已在 only_fine_tune_dec 中隐式实现
        # (Direction 已被冻结)，显式传入 freeze_direction 无意义
        opts.freeze_direction = False
        # beta1 自动设为 0: 无动量，防止小数据上动量累积加速解耦破坏
        if opts.beta1 is None:
            opts.beta1 = 0.0
        # 交叉重建权重自动减半: 少帧时 cross_target 口型差异小，信号弱
        if opts.cross_weight == 0.3:  # 仍为默认值时
            opts.cross_weight = 0.15
        print('==> [auto] only_fine_tune_dec mode: beta1=0.0, freeze_direction=off, cross_weight=0.15 (auto-adjusted)')
    else:
        # 全量微调模式
        if opts.beta1 is None:
            opts.beta1 = 0.5
        print(f'==> [auto] full fine-tune mode: beta1={opts.beta1}, freeze_direction={opts.freeze_direction}')

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
