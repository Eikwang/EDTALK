"""底模行为回归快照（FINE_TUNE_FIX_PLAN T2）。

固定合成输入 → 底模 EDTalk_lip_pose.pt 推理（与实时推理同路径
``Generator.test_from_audio_pose_image``）→ 锁定输出张量统计量。

用途：微调链路（trainer/dataset/损失）的任何代码改动后运行本测试，
保证底模 ckpt 的推理行为不被训练侧改动破坏 — 微调实验的"不劣于现状"
保护网（CEO 评审评为全计划最防后悔项）。

环境钉死（ENG 声部 T2）：快照记录 torch/cuDNN/GPU 指纹；运行环境不符
→ SKIP（跨 torch 版本的浮点漂移会造成误报，不追究）。快照文件不存在
→ 生成基线并 PASS（首次入库由 CI/开发者跑一次即可）。

运行：pytest tests/test_base_model_snapshot.py -v  （需 CUDA）
"""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).parents[1]
SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "edtalk_lip_pose.json"
CKPT_PATH = REPO_ROOT / "ckpts" / "EDTalk_lip_pose.pt"
SEED = 20260904


def _env_fingerprint():
    """当前推理环境指纹 — 与实时推理路径(models.py)保持一致的 TF32 设置。"""
    torch.backends.cudnn.benchmark = False          # 与 models.py.load() 一致
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    props = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
        "cc": f"{props.major}.{props.minor}",
    }


def _build_inputs(device):
    """确定性合成输入：source/pose 驱动帧 + lip alpha（推理分布的形状契约）。"""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    img_source = torch.randn(1, 3, 256, 256, generator=g).to(device)
    pose_drive = torch.randn(1, 3, 256, 256, generator=g).to(device)
    alpha_lip = torch.randn(1, 20, generator=g).to(device)
    return img_source, alpha_lip, pose_drive


@pytest.fixture(scope="module")
def base_model_stats():
    """加载底模并跑一次推理，返回 (环境指纹, 输出统计量 dict)。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA — 快照与实时推理路径同构（GPU 前向），CPU 数值无意义")
    if not CKPT_PATH.exists():
        pytest.skip(f"底模 ckpt 不存在: {CKPT_PATH}")
    sys.path.insert(0, str(REPO_ROOT))

    from networks.generator_lip_pose import Generator

    env = _env_fingerprint()
    device = torch.device("cuda")

    gen = Generator().to(device)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    gen.load_state_dict(ckpt["gen"])  # strict=True — 结构漂移在此直接报错
    gen.eval()

    img_source, alpha_lip, pose_drive = _build_inputs(device)
    with torch.inference_mode():
        out = gen.test_from_audio_pose_image(img_source, alpha_lip, pose_drive)

    assert out.shape == (1, 3, 256, 256), f"意外输出形状: {out.shape}"
    assert not torch.isnan(out).any(), "底模输出含 NaN"
    assert not torch.isinf(out).any(), "底模输出含 Inf"

    stats = {
        "shape": list(out.shape),
        "sum": float(out.to(torch.float64).sum().item()),
        "mean": float(out.to(torch.float64).mean().item()),
        "std": float(out.to(torch.float64).std().item()),
        "min": float(out.min().item()),
        "max": float(out.max().item()),
    }
    return env, stats


def test_base_model_snapshot(base_model_stats):
    """底模输出统计量与快照一致；快照不存在则生成基线。"""
    env, stats = base_model_stats

    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps({"env": env, "stats": stats}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[快照] 基线已生成: {SNAPSHOT_PATH}")
        return

    snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    # 环境不符 → skip 而非 fail（跨 torch/cuDNN/显卡的浮点漂移是预期行为）
    if snap["env"] != env:
        pytest.skip(
            f"环境指纹变化，快照需重建基线:\n  快照: {snap['env']}\n  当前: {env}"
        )

    # 同环境下前向是确定的（cudnn.benchmark=False + eval + inference_mode），
    # 容差只吸收驱动级微差
    for key in ("sum", "mean", "std", "min", "max"):
        base, cur = snap["stats"][key], stats[key]
        denom = max(abs(base), 1e-12)
        assert abs(cur - base) / denom < 1e-6, (
            f"底模行为漂移: {key} 快照={base!r} 当前={cur!r} "
            f"(相对差 {abs(cur - base) / denom:.2e}) — 微调代码改动破坏了底模推理路径"
        )


if __name__ == "__main__":
    test_base_model_snapshot(base_model_stats())
    print("snapshot test passed")
