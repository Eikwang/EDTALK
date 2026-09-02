<!-- /autoplan restore point: /c/Users/admin/.gstack/projects/Eikwang-EDTALK/main-autoplan-restore-20260902-212553.md -->
# 实时推理视线修正集成方案（Gaze Correction Integration）

状态: /autoplan 审查中 | 分支: main | 日期: 2026-09-02

## 1. 问题陈述

视线修正功能在实时管线中完全缺失（PIPELINE_REVIEW.md 问题 2），CLI 仅有一个 placeholder 开关（`--gaze-correction`）。
参考实现 `pytorch_gaze_redirection/src/inference.py` 是离线批处理设计，直接照搬会摧毁实时管线：

| 组件（inference.py） | 耗时特性 | 实时管线处置 |
|---|---|---|
| dlib 全帧人脸检测 + 68 点关键点（每帧） | 10-30ms，CPU | **完全删除**——预处理已提供逐帧 facemark（68 点，256×256 空间），`CharacterDataPack` 启动时已加载 |
| EAR 眨眼状态机（EAR_CLOSE=0.12 / EAR_OPEN=0.16 / 确认 3 帧 / 上限 26 帧） | 可忽略（纯数值） | **原样保留**——EAR 从 facemark 36-41 / 42-47 直接计算 |
| 眼睛中心 EMA 平滑（α=0.4，大位移降权 0.3） | 可忽略 | **原样保留** |
| eye_size 首帧锁定（1.5× 眼眶外包络） | 可忽略 | **保留**，段切换时 reset |
| 眼部裁剪 → PIL + torchvision transforms → 64×64 | 数 ms，PIL 慢 | **替换**：GPU 上直接 slice + `F.interpolate` + 归一化，零 CPU 往返 |
| `generator(batch, angles)` 推理（batch=2，64×64，约 20 层 conv） | 数 ms | **PTH + CUDA Graph**（与主 Generator 同待遇，一次 launch） |
| `skimage.match_histograms` 每帧 2 次 | 5-20ms/次，纯 CPU | **替换**：GPU 单通道 mean/std 颜色迁移（μ/σ 对齐），或先直接禁用观察色差 |
| 圆形 mask 混合贴回 | 数 ms CPU | **替换**：圆形 mask 预计算为 GPU 常量，blend 全程 GPU |

## 2. 管线位置（用户指定 + 架构验证）

```
facemark(68点, 256空间) ──┐
cropped 256×256 (GPU) ────┴─▶ GazeCorrector.correct()
                                   │  (EAR眨眼门控 + 双眼64×64推理 + GPU贴回)
                                   ▼
                          修正后 crop (GPU) ─▶ Audio2Lip/Lip ─▶ Generator ─▶ FaceRepair ─▶ Compositor
                              视线修正 → 口型推理 → 面部修复（用户指定顺序）
```

位置有效性验证：视线修正作用于 **喂给口型 Generator 的 count crop**。修正后的眼睛经 Generator 渲染；
compositor 的嘴部 mask 只贴回嘴区域，眼睛区域的修正结果自然保留在最终帧上。✓

## 3. 设计（GazeCorrector，realtime_serve/gaze_correction.py 新增）

```python
class GazeCorrector:
    """GPU 内联视线修正。眨眼帧自动跳过（EAR 状态机）。"""

    def __init__(self, ckpt_path, device='cuda', convergence_h=0.05, smooth_alpha=0.4):
        # Generator(style_dim=3)，加载 ckpt['current_generator']，eval
        # CUDA Graph 捕获：固定输入 [2, 6, 64, 64]（RGB3 + angles3 tiled）
        # 圆形 mask 常量（64×64，随 eye_size 在 GPU 上 resize）
        # 复刻原版状态：_prev_centers / _fixed_eye_size / blink 计数器

    def reset(self):
        """段切换时调用：清空平滑状态 + eye_size 锁定 + blink 状态。"""

    def correct(self, crop_gpu: torch.Tensor, facemark_256: np.ndarray) -> torch.Tensor:
        """输入输出均为 (3,256,256) RGB float [-1,1] GPU tensor。
        眨眼帧 / facemark NaN：直接返回原 crop（零开销路径）。"""
        # 1. EAR（facemark 36-47，numpy 标量计算 ~0.1ms）
        # 2. 眼中心 EMA 平滑 + eye_size 锁定
        # 3. GPU 裁剪：slice → F.interpolate(64) → 归一化 → stack [2,3,64,64]
        # 4. graph.replay() → 输出 [2,3,64,64]
        # 5. GPU 颜色迁移（μ/σ 对齐替代 match_histograms）
        # 6. GPU resize 回 eye_size + 圆形 mask blend 写回 crop tensor（in-place 或 copy）
```

关键不变式：
1. **输入输出同形状同设备**（3,256,256 RGB [-1,1] GPU）——render_loop 无感知地插在 infer_frame 之前。
2. **眨眼/异常帧零开销旁路**：EAR 高于阈值确认帧直接返回原 crop，不跑模型。
3. **全程 GPU**：crop/推理/贴回无一次 CPU 往返；EAR 与中心平滑用 numpy 标量（facemark 本来在 CPU）。

## 4. 模型格式判断（用户问题：能否转 ONNX）

**结论：可转，全部标准 ONNX op**（Conv / InstanceNorm / ReLU / ConvTranspose / sigmoid / mean-max-cat / tanh，
eval 模式 Dropout 为 no-op，`angles.expand` 在固定 64×64 输入下 trace 无歧义，固定输入 [2,6,64,64]）。

**裁决：仍推荐 PTH + CUDA Graph**（P5 显式优于聪明 / P4 复用）：
- models.py 已有完整的 Graph 捕获基建（主 Generator 刚验证过），gaze Generator 照搬 ~20 行；
- 零转换脚本、零精度验证负担、零新文件格式；
- ONNX 导出脚本作为 TODOS 后备项：仅当部署目标脱离 PyTorch 运行时才启用。

## 5. render_loop 集成（T2）

```python
# _process_frame()，Generator 推理之前
cropped_gpu = cropped  # (1,3,256,256) GPU，已含 normalize
if self.gaze_corrector is not None and facemark is not None:
    cropped_gpu = self.gaze_corrector.correct(cropped_gpu[0], facemark).unsqueeze(0)
rendered = self.model_manager.infer_frame(cropped_gpu, alpha_D_lip, cropped_gpu)
```

- `switch_segment()` → `gaze_corrector.reset()`；
- 主 Generator 的 CUDA Graph 输入形状不变（256×256 恒定）——不受影响；
- timing 直方图新增 `gaze` 阶段。

## 6. 配置与可观测性（T3/T4）

- `--gaze-correction` 升级为三态：`off`（默认）/ `on` / `adaptive`（复用 face_repair 的 `_repair_allowed()` 语义：buffer 深度 ≥ REPAIR_MIN_BUFFER 才启用）；
- `/status` 增加 `gaze_state: off|on|adaptive|blink_skip`；
- RUNBOOK.md 增补：参数行 + 眨眼跳过说明 + ONNX 备选路径说明。

## 7. 性能预算（验收依据）

当前产能富余（log13 后续 CUDA Graph 运行）：gen 8.9 + repair 7.2 + comp 0.3 + put 背压 20 = total 39.7ms
→ 实际生产 ≈ 19.7ms/帧 ≈ **50fps 产能**。

| 新增项 | 预算 |
|---|---|
| EAR + 中心平滑（CPU numpy） | ~0.2ms |
| GPU crop + interpolate + normalize | ~0.5ms |
| gaze Generator graph replay（64×64×2） | ~1-2ms |
| GPU 颜色迁移 + mask blend | ~0.5ms |
| **合计** | **~2-4ms/帧** |

预算后产能 ≈ 40fps，仍远超 25fps 输出线。验收指标：`--gaze-correction on` 下 FPS 25 稳定、buffer 蓄满、
`gaze` p99 < 10ms、眨眼帧不修正（日志计数）。

## 8. 已知代价与风险

- **色差风险**：match_histograms 被替换为 μ/σ 迁移，生成眼睛与眼周色域可能出现轻微差异。缓解：先 on 模式目检；
  若不可接受再评估 GPU 直方图匹配（复杂度高，暂不入 scope）。
- **facemark 质量依赖**：闭眼/侧脸帧 facemark 可能漂移 → EAR 误判。缓解：复用原版状态机的滞回 + 上限帧数，
  误判最坏后果 = 该帧不修正（安全方向）。
- **CUDA Graph 捕获失败**：回退 eager（复用主 Generator 的 fallback 模式），gaze 推理变慢但不崩。

## 9. 明确不在本次范围

- gaze Generator 的 ONNX 导出脚本（可转已验证，仅在脱离 PyTorch 部署时启用）→ TODOS
- GPU 直方图匹配（μ/σ 迁移不达标时的后备）→ TODOS
- 主 Generator 重训练 / 头部姿态修正（正交）

## 10. 复用清单（What already exists）

| 需求 | 已有资产 |
|---|---|
| 逐帧 68 点 landmarks（256 空间） | CharacterDataPack._segment_cache['facemark']（启动加载） |
| CUDA Graph 捕获/回退模式 | models.py._capture_generator_graph（照搬） |
| adaptive 门控语义 | render_loop._repair_allowed()（同模式新增） |
| 三态 CLI 模式 | --face-repair on/adaptive 解析（同模式） |
| timing 直方图 | render_loop._timings（加 gaze 键） |
| 眨眼状态机参数 | inference.py 常量（原样复刻） |
