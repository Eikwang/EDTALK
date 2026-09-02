# EDTalk 实时推理服务运行手册

## 重要语义变更（2026-09-02）

`--target-fps` 现在**只控制最终向虚拟摄像头的输出帧率**，不再限制渲染生产速度。

- 旧行为：`RenderLoop` 以 target_fps 节拍生产并直接发送。
- 新行为：`RenderLoop` 全速渲染入 `FrameBuffer`；`OutputStreamer` 以 target_fps 消费并输出。
- 结果：输出更稳定，角色动作速度不再随生产抖动而变（不再"快放"）。

## Generator CUDA Graph

启动时 Generator 前向被捕获为 CUDA Graph（日志 `==> Generator CUDA Graph captured (fp32)`）。每帧推理从上百次逐层 kernel launch 变为一次图重放，消除 GPU 层间空转。捕获失败自动回退 eager 推理（日志 `capture failed`）。要求输入形状固定 1×3×256×256（本服务恒定满足）。

## 启动参数

| 参数 | 默认值 | 单位 | 说明 |
|------|--------|------|------|
| `--target-fps` | 25 | fps | 仅约束摄像头输出帧率 |
| `--buffer-frames` | 16 | 帧 | 输出缓冲容量；每帧约 11MB @ 1440×2560。加大可吸收更长尖峰串，但不能弥补平均产能不足 |
| `--preroll-frames` | 8 | 帧 | 启动/新段后需预填充的帧数；必须 ≤ buffer-frames |
| `--batch-size` | 100 | 帧 | 角色素材预加载批次 |
| `--no-face-repair` | 默认 | - | 关闭面部修复（推荐，输出最流畅） |
| `--face-repair on` | - | - | 每帧修复。已改为 **GPU 内联（IOBinding 零拷贝）**，消除了每帧两次 GPU↔CPU 往返；但 GPU 推理本身 ~6-7ms 仍在，产能紧张时仍建议关闭 |
| `--face-repair adaptive` | - | - | 自适应：仅当输出缓冲深度 ≥6 帧（产能有余量）时修复，紧张时自动跳过。GPU 内联模式下推荐此项 |
| `--no-unitycapture` | - | - | 无虚拟摄像头时仍可运行；OutputStreamer 正常 pacing |
| `--gen-fp16` | 关闭 | - | 开启 Generator fp16 推理。**默认关闭**：实测在 RTX 4080 SUPER 上慢于 fp32+TF32（TF32 已覆盖 Ada 的大部分半精度收益，autocast 只剩 cast 开销） |

### 参数换算

- `buffer-frames=16`（默认）→ 约 176MB 内存，吸收深度 0.64s。
- `buffer-frames=8` → 约 88MB，吸收深度 0.32s。
- `buffer-frames=300`（上限）→ 约 3.3GB，吸收深度 12s（不推荐：只对"平均产能 ≥ 25fps"才有意义）。

## 状态字段（GET /status）

| 字段 | 含义 |
|------|------|
| `production_fps` | 渲染生产速度 |
| `output_fps` | 摄像头输出目标帧率 |
| `fps` | `output_fps` 的兼容别名（已弃用） |
| `buffer_depth` | 当前缓冲深度 |
| `underruns` | 输出端欠载重发计数 |
| `preroll_state` | `prerolling` / `complete` |
| `buffer_full_warn` | 缓冲持续满（检查摄像头消费端） |

## 常见问题与处置

### 画面卡顿 / underruns 持续增加

1. 看 `production_fps`：
   - 若长期 < `output_fps` → 产能不足，考虑关闭 `--no-face-repair` 或降低 `--target-fps`。
   - 若 `production_fps` 正常但 `underruns` 增加 → 检查 UnityCapture / OBS 是否消费变慢。
2. 临时降级：设置 `--buffer-frames=1` 回到近旧行为。

### 启动后摄像头黑屏数秒

这是正常的 preroll 等待。默认 4 帧 ≈ 0.16s。需要更快启动可设 `--preroll-frames=0`（但会牺牲启动平滑度）。

### 推音频后嘴型延迟

- 阶段 1 已做缓冲深度前视补偿 + DAC 延迟补偿。
- 若仍感觉不同步，检查 `AudioPlayer.output_latency_s` 是否被正确估计。

### 无摄像头模式测试

```bash
python -m realtime_serve.main --character-dir test_data/wy --no-unitycapture
```

输出 pacing 和音频触发仍正常工作，只是不向摄像头送帧。

### 快速开始（跳过 preroll 的测试捷径）

```bash
python -m realtime_serve.main --character-dir test_data/wy --preroll-frames=0
```

## 回滚

本阶段改动集中在 `realtime_serve/output_buffer.py` 和渲染管线。若需要回退，可用 git revert 最近一次提交。
