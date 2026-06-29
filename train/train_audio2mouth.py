# 解决 Windows OpenMP 冲突（必须在任何 import 之前设置！）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CUDA 内存分配策略:减少 Windows 上的内存碎片导致的 access violation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import sys

# 确保项目根目录在 sys.path 中（支持从任意目录运行）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import datetime
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

# 不设置 set_per_process_memory_fraction 硬限制,避免 backward 峰值撞上限制导致
# Windows WDDM 静默崩溃。改用显式 del + empty_cache 控制显存。

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")

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
    dataset_test = Audio2LipDataset_image_sync(hdtf=args.data_path, is_train=False, transform=transform,
                                                preload_features=args.preload_features)

    # 限制测试集为训练集的 1%, 避免每次 eval 跑一整轮 (与训练同规模) 浪费时间
    # 原因: train/test 共享同一份 samples 列表, eval 一次要跑 2 万+ batch, ~1 小时
    test_size = max(1, len(dataset.samples) // 100)
    dataset_test.samples = dataset_test.samples[:test_size]
    print(f"[Dataset] 测试集截断: {len(dataset_test.samples)} 样本 (训练集 1%, "
          f"原 {len(dataset.samples)} 条)")
    # elif args.dataset == 'taichi':
    #     dataset = Taichi('train', transform, True)
    #     dataset_test = Taichi('test', transform)
    # else:
    #     raise NotImplementedError

    if args.distributed == False:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True, drop_last=True,
                                persistent_workers=True, prefetch_factor=2)
        # 关键修复: test_dataloader 不使用多进程 worker
        # 原因: Windows spawn 模式下 persistent_workers=True 会导致 eval 时 worker 死锁
        # (worker 在训练阶段长时间空闲后 LMDB env 失效, 主进程永远阻塞)
        # 验证集不大, 主进程加载完全够用, 彻底消除死锁风险
        test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,
                                     num_workers=0, pin_memory=True, drop_last=True)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                                num_workers=2, sampler=train_sampler, pin_memory=True, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=(test_sampler is None), 
                                     num_workers=2, sampler=test_sampler, pin_memory=True, drop_last=True)

    trainSteps = len(dataloader)
    testSteps = len(test_dataloader)
    # loader = sample_data(loader)
    # loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt, args.audio2lip_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    current_iter = args.start_iter
    test_iter = 0
    print('==> training')
    # pbar = range(args.iter)
    last_name = None
    test_epoch= 0
    for epoch in range(args.epoch):
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
                    total=trainSteps, leave=True, ncols=100)
        for bii, bi in enumerate(pbar):
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
                accumulation_steps=args.accumulation_steps,
                step_optimizer=step_optimizer
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
                    with torch.no_grad():
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
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--size", type=int, default=256)
    
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    
    parser.add_argument("--lr", type=float, default=0.0005)
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
    parser.add_argument("--image_save_iter", type=int, default=2500, help="local rank for distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--distributed", action='store_true', help="Enable distributed training")
    parser.add_argument("--eval_iter", type=int, default=10000, help="local rank for distributed training")
    parser.add_argument("--sync_weight", type=int, default=0, help="SyncNet loss weight (0=disabled, need syncnet_ckpt)")
    parser.add_argument("--syncnet_ckpt", type=str, default=None, help="Path to SyncNet pre-trained checkpoint (e.g. ckpts/EDTalk.pt)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dis_weight", type=int, default=0, help="GAN 判别器权重 (0=禁用)。微调推荐关闭：小数据集 GAN 易崩溃，且 d_optim 链路未启用时 dis 仅向 audio2lip 注入噪声梯度")
    parser.add_argument("--vis_num", type=int, default=10, help="local rank for distributed training")
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--train_generator", type=bool, default=False, help="local rank for distributed training")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--audio2lip_ckpt", type=str, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数（单卡显存不足时使用，等效 batch_size = batch_size * accumulation_steps）")
    parser.add_argument("--data_path", type=str, default='HDTF', help="Dataset base directory path")
    parser.add_argument("--log_freq", type=int, default=5000, help="每 N 个 batch 打印一次滚动日志（0=只在 epoch 结束打印）")
    parser.add_argument("--preload_features", action='store_true',
                        help="将所有特征预加载到内存 (小数据集微调推荐, 大数据集慎用)")
    
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