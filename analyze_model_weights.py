"""
对比 EDTalk_lip_pose.pt 和 EDTalk.pt 的权重层结构
输出: 两个模型的所有权重键及形状, 以及差异对比
"""
import torch
import sys
from collections import defaultdict

def inspect_model(path, label):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'gen' not in ckpt:
        print(f'{label}: 没有 gen 键')
        return set()
    gen_keys = list(ckpt['gen'].keys())
    print(f'=== {label}: {path} ===')
    print(f'  Total keys: {len(gen_keys)}')

    # Group by module prefix
    groups = defaultdict(list)
    for k in gen_keys:
        prefix = '.'.join(k.split('.')[:2])
        groups[prefix].append(k)

    print(f'  Module groups:')
    for g in sorted(groups.keys()):
        print(f'    [{g}] ({len(groups[g])} keys)')
    print()

    # Print full list with shapes
    print(f'  Full weight list:')
    for k in gen_keys:
        shape = tuple(ckpt['gen'][k].shape)
        print(f'    {k}: {shape}')
    print()
    return set(gen_keys)


def inspect_extra_keys(path, label):
    """检查是否有dis, optim等其他键"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f'=== {label}: extra keys in checkpoint ===')
    for k in ckpt.keys():
        if isinstance(ckpt[k], dict):
            print(f'  [{k}]: dict with {len(ckpt[k])} sub-keys')
            # print some sub-keys
            sub_keys = list(ckpt[k].keys())[:10]
            for sk in sub_keys:
                try:
                    shape = tuple(ckpt[k][sk].shape)
                    print(f'    {sk}: {shape}')
                except:
                    pass
        else:
            print(f'  [{k}]: {type(ckpt[k])}')
    print()


if __name__ == '__main__':
    lp_path = r'd:\AI\EDTalk\ckpts\EDTalk_lip_pose.pt'
    full_path = r'd:\AI\EDTalk\ckpts\EDTalk.pt'

    inspect_extra_keys(lp_path, 'EDTalk_lip_pose.pt')
    inspect_extra_keys(full_path, 'EDTalk.pt')

    keys_lp = inspect_model(lp_path, 'EDTalk_lip_pose.pt')
    keys_full = inspect_model(full_path, 'EDTalk.pt')

    # 差异对比
    print('='*80)
    print(f'Keys ONLY in EDTalk_lip_pose.pt (简化版特有):')
    only_lp = sorted(keys_lp - keys_full)
    if only_lp:
        for k in only_lp:
            print(f'  {k}')
    else:
        print('  (无)')
    print()

    print(f'Keys ONLY in EDTalk.pt (完整版特有 - 表情控制相关):')
    only_full = sorted(keys_full - keys_lp)
    if only_full:
        for k in only_full:
            print(f'  {k}')
    else:
        print('  (无)')
    print()

    # 共享键的形状对比
    print('='*80)
    print(f'Shared keys with DIFFERENT shapes:')
    ckpt_lp = torch.load(lp_path, map_location='cpu')
    ckpt_full = torch.load(full_path, map_location='cpu')
    common = keys_lp & keys_full
    diff_count = 0
    for k in sorted(common):
        s1 = tuple(ckpt_lp['gen'][k].shape)
        s2 = tuple(ckpt_full['gen'][k].shape)
        if s1 != s2:
            print(f'  {k}: lip_pose={s1} vs full={s2}')
            diff_count += 1
    if diff_count == 0:
        print('  (无差异 - 所有共享键的形状完全一致)')
    print()

    print('=== 总结 ===')
    print(f'  EDTalk_lip_pose.pt: {len(keys_lp)} 权重')
    print(f'  EDTalk.pt: {len(keys_full)} 权重')
    print(f'  仅在简化版: {len(keys_lp - keys_full)}')
    print(f'  仅在完整版: {len(keys_full - keys_lp)}')
    print(f'  共享且形状不同: {diff_count}')
