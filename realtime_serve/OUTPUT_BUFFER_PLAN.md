<!-- /autoplan restore point: ~/.gstack/projects/EDTalk/main-autoplan-restore-20260902.md -->
# 实时推理输出缓存池方案（Output Buffer Pool）

状态: /autoplan 审查中 | 分支: main | 日期: 2026-09-02

## 1. 问题陈述

帧率在 18-25 之间波动，画面不稳定。实测证据：

1. **生产能力充足**：不加 pacing 时生产可达 30+fps，CPU/GPU 占用仅 ~25%。
   帧率不稳定不是性能不足，是代码逻辑问题。
2. **"快放"现象**：`--target-fps` 同时控制视频时钟
   （render_loop.py:407 `int(elapsed * target_fps) % total`）。序列帧素材是
   25fps，时钟跑 30fps 时人物动作 1.2 倍速。运动速度与输出 pacing 耦合。
3. **生产抖动被直接暴露给摄像头**：生成单帧 20-50ms 波动（log5），超过 40ms
   预算的帧导致发送端 latest-wins 槽位重复旧帧 → 可感知卡顿。
4. **生产侧 sleep pacing 精度差**：`time.sleep(wait*0.8)` 受 Windows 调度
   量子影响，pacing 本身引入抖动。

## 2. 方案（用户提议 + 本方案细化）

生产-消费解耦：

- **生产环节（RenderLoop）不限速**：尽快渲染，全部进入缓存池。
- **输出环节（UnityCapture 发送线程）以 target-fps 精确 pacing**：从缓存池
  取帧送虚拟摄像头。
- **缓存池大小可配置**（如 50 帧 = 2 秒），启动参数控制；预填充达到阈值后
  才开始向摄像头推流。
- **`--target-fps` 语义变更**：只限制最终摄像头输出帧率，不再参与生产节拍。

## 3. 细化设计（可行性结论：可行，需以下配套改动）

核心不变式（缺一则唇音失步）：

1. **唇形/背景帧选择从"墙钟"改为"序列号"**。
   生产帧携带单调递增 seq：`video_idx = (video_offset + seq) % total`；
   音频激活时 `lip_idx = seq - audio_base_seq`。输出端按 target-fps 逐帧
   消费，第 n 帧在音频起点后 n/fps 秒显示，其 lip_idx 恰为 n → 唇音同步
   由构造保证，与生产速度无关。
2. **音频播放由输出线程触发**：段首帧打上 `audio_start` 标记，输出线程
   pop 到该帧的时刻 push PCM → 音频起点 = 该帧上屏时刻，精确对齐。
   （现状是第一块 lip 算完立即 push，与缓冲解耦后必须迁移触发点。）
3. **背压**：缓存池满时生产者阻塞 → 稳态生产率 = 输出率 = 25fps，
   GPU 只在填池阶段全速运转。
4. **欠载策略**：缓存池空时输出线程重发上一帧（摄像头保持 25fps 节奏），
   重新武装 preroll，计数并在 /status 暴露 underrun。
5. **新段推送时 flush 缓存池**：丢弃未上屏的 idle 帧，段间延迟
   ≈ 首块 lip 计算时间 + ≤1 帧。
6. **Windows 定时器精度**：启动时 `timeBeginPeriod(1)`（ctypes），
   输出线程用 perf_counter 锚点 + 混合 sleep/busy-wait。
7. **漂移校正**：每 ~2s 对比 AudioPlayer 已耗样本时间 vs 已 pop 帧数，
   |漂移| > 80ms 时平移 audio_base_seq（音频硬件时钟与 QPC 的 ppm 级漂移）。

## 4. 落地文件

| 文件 | 改动 |
|------|------|
| `realtime_serve/output_buffer.py` | 新增 FrameBuffer（有界 FIFO、Condition、flush、preroll 事件、统计） |
| `realtime_serve/unity_capture.py` | 发送线程 latest-wins → FIFO 消费 + preroll + audio_start 钩子 + underrun 统计 |
| `realtime_serve/render_loop.py` | 生产循环去 pacing；墙钟→序列号；buffer.put；新段 flush；漂移校正 |
| `realtime_serve/main.py` | 新增 `--buffer-size`（默认 50）、`--preroll`（默认=buffer-size）；timeBeginPeriod |
| `realtime_serve/config.py` | buffer_size / preroll 字段 |
| `realtime_serve/api/app.py` | /status 增加 buffer_depth、underrun、output_fps、production_fps |
| `tests/` | 新增 test_output_buffer.py、发送线程测试；更新 render_loop/unity_capture 测试 |

## 5. 已知代价与风险

- **内存**：50 帧 × 11MB（1440×2560×3 BGR）≈ 550MB RAM。可通过 --buffer-size 降。
- **启动延迟**：preroll=50 → 服务启动后 ~2s 才开始推流（一次性）。
- **持续欠产能**：生产长期 <25fps 时缓存池会周期性欠载（冻结-回填循环）；
  以 underrun 统计暴露。生产实测 30+fps，正常余量充足。
- **无摄像头模式**（--no-unitycapture）：生产无限速会空转烧 GPU，
  需退化为 target-fps 自限速。

## 6. 明确不在本次范围

- Generator 模型级优化（TensorRT / fp16 / 批量推理）——正交，后续项。
- 视线纠正（占位接口保留）。
- 前端界面（无）。

---

# CEO 审查（Phase 1）— 模式: SELECTIVE EXPANSION

## 系统审计

- 连续 8 个 perf/fix 提交集中在渲染热路径（d393665→0f414fa）：反复修整的区域，
  按回顾性检查原则从严审查。
- 无 stash、无 TODO/FIXME 遗留、无历史 learnings。
- 仓库内已有设计材料：specs/features/realtime-streaming-inference_*.md（跳过
  /office-hours 前置提议，理由：设计上下文已在 repo 且本次为延续性工程决策）。

## 0A. 前提挑战

| # | 前提 | 评估 | 结论 |
|---|------|------|------|
| P1 | 性能充足（~25% 占用），帧率不稳是逻辑问题 | 基本成立：gen=20-50ms 波动 + sleep pacing 抖动 + latest-wins 透传。但 gen 波动根因未查明，方案是"解耦"而非"消除" | 接受，附条件：必须加 underrun 遥测验证 |
| P2 | 缓存池可稳定 25fps 输出 | **只在生产平均 ≥ 输出率时成立**。缓存吸收抖动，不弥补持续亏空 | 接受，写入方案显式条件 |
| P3 | （用户未提及）墙钟唇同步必须改为序列号 | 缺此改动缓冲直接破坏唇音同步 | 接受，为必要配套改动 |
| P4 | （用户未提及）音频起点必须迁到输出线程 | 同上 | 接受 |
| P5 | 550MB 内存可接受 | 工作站 + RTX 环境，合理 | 接受，buffer-size 默认值留门上裁决 |
| P6 | --target-fps 语义变更是破坏性变更 | 操作者即用户本人，可控 | 接受 |

"什么都不做"的代价：卡顿持续存在，且 gen 波动根因（GPU 侧）短期无法消除。
真实痛点，非假设。继续。

## 0B. 现有代码杠杆

| 子问题 | 现有代码 | 复用决策 |
|--------|----------|----------|
| 有界帧队列 | 无（unity_capture 只有 latest-wins 单槽） | 新建，但基于 stdlib deque+Condition 薄封装，不造轮子 |
| 输出端 fps pacing | UnityCaptureSender._sender_loop 已有 perf_counter 锚点 pacing + 饥饿重发 | **直接复用**该循环骨架，只改槽语义为 FIFO |
| 视频时钟 | render_loop._video_idx 墙钟 | **替换**为序列号（这正是修复本身） |
| 音频起点 | process_full_audio 首块就绪即 push | **迁移**触发点到输出线程 pop 侧 |
| idle 唇形缓存 | _idle_lip_cache | 原样复用 |
| 状态监控 | /status 端点 | 扩展字段 |
| Windows 定时器精度 | 无 | 新增 timeBeginPeriod(1)（一处 ctypes 调用） |

无重复建设。发送线程 pacing 骨架是最有价值的存量资产。

## 0C. 梦想状态映射

```
CURRENT STATE                    THIS PLAN                        12-MONTH IDEAL
生产25fps(sleep pacing)   --->   生产不限速→FIFO(50,preroll) --->  生成侧批量化/TensorRT
latest-wins→发送25fps            →发送25fps(序列号唇同步)          (gen<10ms)
唇形=墙钟(双时钟)                音频起点=pop时刻(构造保证)        自适应preroll+帧级metrics
                                 +漂移校正                          多角色热切换
```

方向判断：**靠近理想态**。解耦是生成侧后续优化的前提——先让输出与生产速度
无关，之后 gen 提速才转化为余量而非提速。

## 0C-bis. 实现备选方案

```
APPROACH A: 仅修 pacing 精度（timeBeginPeriod + busy-wait，保留墙钟）
  Summary: 最小改动，去 sleep 抖动，摄像头仍按墙钟取帧
  Effort:  S    Risk: 低
  Pros:    1天内完成；零新概念；无内存代价
  Cons:    GPU 20-50ms 抖动直接透传，>40ms 帧仍被摄像头重复；不满足目标
  Reuses:  现有 sender 循环
  Completeness: 5/10（捷径，不达标）

APPROACH B: 缓存池方案（用户提议 + 序列号细化）★
  Summary: 生产不限速入 FIFO；输出线程 25fps 消费；唇同步/音频起点随迁
  Effort:  M（human ~2天 / CC ~1小时）   Risk: 中
  Pros:    彻底解耦；唇同步由构造保证；运动速度与输出率解耦（"快放"根除）
  Cons:    550MB 内存；启动 preroll 延迟；新增并发组件需测试
  Reuses:  sender pacing 骨架、idle 缓存、/status 端点
  Completeness: 9/10

APPROACH C: 批量推理（Generator 一次产 2-4 帧）摊薄单帧开销
  Effort:  L    Risk: 高
  Pros:    从根上提高产能下限
  Cons:    模型层大改；与 lip chunk 渐进填充管线冲突；端到端延迟上升
  Reuses:  无
  Completeness: 7/10（但方向不同，属模型优化而非架构修复）
```

**裁决（P1 最高完整性）：选 B。** A 不达标（完整性 5/10 直接排除）；
C 是正交的后续项 → TODOS。A 中的 timeBeginPeriod(1) 被吸收进 B。

## 0D. SELECTIVE EXPANSION 分析

复杂度检查：6-7 文件、2 个新组件（FrameBuffer + 发送线程改造）→ 低于 8 文件
异味阈值。通过。

最小改动集：缓冲池 + 序列号时钟 + 音频起点迁移。三者缺一不成立，不可拆分延迟。

扩展候选（autoplan 自动裁决）：
| # | 提案 | 裁决 | 理由 |
|---|------|------|------|
| E1 | 时钟漂移校正（音频硬件时钟 vs QPC，>80ms 平移 audio_base_seq） | 采纳 | 爆炸半径内、<1 天、唇同步长期正确性的必要件 |
| E2 | /status 扩展（buffer_depth/underrun/双 fps/preroll 状态/drift_ms） | 采纳 | 可观测性是 scope（Prime Directive 5） |
| E3 | underrun 后自适应 preroll（动态调阈值） | TODOS(P3) | 需要运行数据才设计得出，先遥测 |
| E4 | 帧时间直方图 metrics | TODOS(P3) | 同上 |

Delight 项：启动时回显 buffer/preroll 配置；underrun 日志附 buffer 深度轨迹
→ 采纳（S 工作量，半径内）。

## 0E. 时序审讯

- HOUR 1: video_offset 的段切换语义（= 切换时 video_idx - seq，保证视觉连续）；
  Python int 无溢出问题。
- HOUR 2-3: 歧义——flush 与 audio_start 标记帧的相互作用（定死：新段 push 时
  重置 pending audio 状态，标记帧必为新段首帧）；stop() 时 buffer.put 阻塞
  死锁（put 循环带 timeout 检查 _running）。
- HOUR 4-5: 意外——AudioPlayer.push_pcm 在输出线程调用的线程安全性（内部有
  lock，安全）；preroll 等待期 /status 需显示 "pre-rolling"。
- HOUR 6+: 测试——实时 pacing 测试在 CI 不稳 → 时钟注入（SenderClock 可替换），
  单测零真实 sleep。

## 0F. 模式确认

SELECTIVE EXPANSION（autoplan 覆盖），方案 B 落位。E1/E2/delight 项并入 scope，
E3/E4 → TODOS。

## 11 节深度审查

### S1 架构审查

```
        ┌────────────────────────────────────────────────────────────┐
        │                    改造后数据流                              │
        │                                                            │
 API ──▶ process_full_audio          RenderLoop._run（生产，不限速）    │
 │        (chunked Audio2Lip)          seq+=1                         │
 │        不再 push PCM                video_idx=(offset+seq)%total   │
 │        记 pending_pcm               lip: 音频? seq-base_seq        │
 │                                     : idle缓存                     │
 │                                     gen→post→repair→composite      │
 │                                            │ put(帧,seq,标记)       │
 │                                            ▼ [阻塞当满]            │
 │                                     ┌──────────────┐               │
 │                                     │ FrameBuffer  │ FIFO+preroll  │
 │                                     └──────┬───────┘               │
 │                                            │ get(timeout)          │
 │                                            ▼                       │
 │        AudioPlayer ◀── push_pcm ── 输出线程(25fps pacing):        │
 │          (独立播放)            pop→遇段首帧:push_pcm+记起点        │
 │                                → virtual_cam.send()                │
 │                                → 空则重发上帧(underrun++)           │
 └── flush 触发: 新音频 push 时清空未播帧 + 重置 pending 状态          │
```

- 耦合变化：render_loop ↔ output_buffer（新增），输出线程 → AudioPlayer
  （新增回调）。裁决：回调注入（显式优于聪明），发送线程不 import 音频模块。
- SPOF：输出线程崩溃 = 无输出。修复：循环体 try/except + 退避 + 错误计数。
  已并入任务 T4。
- 回滚姿态：git revert 单提交即可，无迁移、无持久化格式变更。二类门。
- 扩展性：10x 负载 = 更多并发 API 推送，与渲染管线无争用（_push_lock 已有）。
  100x 同理，瓶颈仍在 GPU。

### S2 错误与救援映射

```
METHOD/CODEPATH            | 可出错场景                  | 异常类
---------------------------|-----------------------------|------------------
FrameBuffer.put            | 满且 stop() 竞争            | (timeout循环检查_running)
FrameBuffer.get            | 空且等待                    | Empty(timeout返回None)
sender loop: send          | Unity 不消费/共享内存满      | Exception(现有处理)
sender loop: audio_start_cb| 回调抛异常                  | Exception ← 新增
process_full_audio chunk   | GPU 推理异常（中段失败）      | RuntimeError
_composite/_gen（生产侧）   | 单帧异常                    | 各类
```

```
异常                        | 救援? | 动作                       | 用户所见
----------------------------|-------|----------------------------|------------------
send 失败                    | Y     | 计数+日志（现有）           | 画面保持上帧
audio_start_cb 异常          | Y(新增)| 捕获+计数+日志，不影响发送  | 无（音频起点缺失会记 underrun 级告警）
chunk 中段失败               | Y     | lip_filled 停在已完成行，   | 嘴型定格在最后完成帧
                            |       | 生产侧 clamp（现有模式复用）|
FrameBuffer stop 竞争        | Y     | put 带 timeout 循环        | 服务正常退出
```
无 catch-all 新增。无 CRITICAL GAP。

### S3 安全与威胁模型

无新端点、无新外部输入（CLI int 由 argparse + 范围校验：buffer_size≥1,
preroll≥1）。无密钥、无依赖新增。/status 暴露内部指标——LAN 服务，与现状
一致。无注入面。无发现。

### S4 数据流与边界用例

| 交互 | 边界用例 | 处理 |
|------|----------|------|
| 新音频 push | 空 PCM | 现有守卫（empty audio ignored） |
| 新音频 push | 音频比 preroll 短 | preroll=min(preroll, total_lip_frames)，生产持续 hold 帧最终填满 |
| 新音频 push | 上一段还在播 | flush 未播帧 + 重置 pending（现行为等价：旧段被切断） |
| preroll 等待中 | stop() | event wait 带 timeout 循环 |
| 播放中 | buffer 空持续 >2s | 重发上帧 + underrun++ + 重武装 preroll |
| 长时间运行 | 音频钟 vs QPC 漂移 | E1 漂移校正（>80ms 平移） |
| 段切换 | video_offset 捕获 | 切换时 offset = 当前video_idx - seq，视觉无跳变 |

### S5 代码质量

FrameBuffer 预计 ~80 行，deque+Condition 薄封装，无过度设计。复用 sender
pacing 循环（DRY）。unity_capture.send_frame 同步模式保留（测试兼容）。
无圈复杂度超 5 的新方法。无发现。

### S6 测试图

```
新代码路径:
  FrameBuffer     : FIFO序/满阻塞/flush/preroll事件/统计     → 单测（假线程，无sleep）
  输出线程 pacing : 25fps节奏/空重发/underrun计数            → 单测（注入时钟）
  audio_start回调 : pop段首帧触发一次/flush后不误触发         → 单测
  序列号时钟      : video_idx回绕/lip clamp/渐进填充clamp     → 单测
  漂移校正        | >80ms平移/<80ms不动                       → 单测（假时钟）
  集成            : 生产慢3s→underrun→恢复                   → 集成（真实线程，容忍±）
更新的既有测试:
  test_t5_render_loop (去pacing断言/序列号断言)
  unity_capture 测试 (latest-wins→FIFO)
```
凌晨 2 点放心发布的测试：集成 underrun-恢复循环。敌意 QA：put 与 flush 并发
竞态、stop 时三线程汇合。混沌：生产者随机 sleep 模拟 GPU 抖动。无时间敏感
断言（时钟注入）。金字塔：多单测、少集成、无 E2E。

### S7 性能

内存峰值 550MB@50 帧（已文档化，--buffer-size 可调）。put/get O(1)。
输出线程预算：pop(≈0)+send(5-10ms) << 40ms。生产稳态被背压钳在 25fps，
GPU 只在填池期全速。p99(pop+send)≈12ms。无 N+1 类问题（无 DB）。无发现。

### S8 可观测性

/status 新增：production_fps、output_fps、buffer_depth、underruns、
preroll_state、drift_ms（E2，已并入 scope）。日志：underrun 附深度轨迹、
audio_start 延迟、flush 帧数、启动配置回显（delight 项）。运行手册：持续
underrun → 降 target-fps 或关 face_repair。控制台 5s 周期打印（沿用现有模式）。

### S9 部署与回滚

本机服务无迁移。回滚 = git revert。CLI 参数即特性开关（--buffer-size 1 =
近浅缓冲退化模式）。部署后验证清单：启动→preroll 完成→idle 25fps 稳定→
推音频→唇音同步目检→/status 字段齐全→underrun=0。无风险窗口（单实例）。

### S10 长期轨迹

可逆性 4/5（revert 即回，无格式/数据变更）。技术债：双墙钟概念被序列号
**简化**（减债）；漂移校正新增小复杂度（有测试覆盖）。路径依赖：为批量
推理/TensorRT 铺路——解耦后生成侧优化纯转化为余量。12 个月后新工程师
读图即懂。无发现。

### S11 设计与 UX — 跳过（无 UI 范围）

## NOT in scope（含理由）

- 批量推理/TensorRT/fp16：模型层正交优化，等缓冲池落地后按余量数据决策 → TODOS
- 自适应 preroll：需运行遥测才能设计 → TODOS
- 帧时间直方图：同上 → TODOS
- 视线纠正实现：占位接口既有，独立特性
- 多角色热切换：未在本次问题域内

## What already exists（复用清单）

见 0B 表。关键复用：sender pacing 循环骨架（unity_capture.py:149-169）、
idle 唇缓存（render_loop._get_idle_lip_coeff）、渐进填充 clamp 模式
（render_loop.py:346-349）、/status 端点（api/app.py:145）、_push_lock
推送串行化（render_loop.py:116）。

## 梦想状态差值

本方案落地后：输出稳定性与生产波动完全解耦（理想态的核心地基已完成）；
距 12 个月理想态剩余：生成侧提效（C 方案）、自适应 preroll（E3）、
metrics 面板（E4）——均有明确路径，无架构性障碍。

## Failure Modes Registry

```
CODEPATH            | FAILURE MODE         | RESCUED? | TEST? | USER SEES?      | LOGGED?
--------------------|----------------------|----------|-------|-----------------|--------
输出线程 send        | Unity不消费          | Y        | Y     | 画面保持上帧     | Y
输出线程 audio_start_cb| 回调异常            | Y(新)    | Y     | 无直接感知       | Y+计数
生产循环单帧异常     | gen/comp 抛异常      | 部分(现状)| Y     | 上一帧重复       | Y
FrameBuffer 满+stop  | 死锁风险             | Y(新)    | Y     | 服务正常退出     | N/A
buffer 持续空        | 产能不足             | Y(新)    | Y     | 冻结+重武装      | Y+计数
音频漂移             | 时钟 ppm 级偏差      | Y(E1)    | Y     | 无（<80ms）      | Y(drift_ms)
```
CRITICAL GAP: 0

## Implementation Tasks（CEO 阶段）

- [ ] **T1 (P1, human: ~4h / CC: ~20min)** — output_buffer — 新建 FrameBuffer
  - Surfaced by: 0C-bis 方案B / S1 架构图
  - Files: realtime_serve/output_buffer.py（新增）
  - Verify: pytest tests/test_output_buffer.py
- [ ] **T2 (P1, human: ~6h / CC: ~30min)** — render_loop — 生产循环去 pacing +
  序列号时钟（video_offset/lip clamp/渐进填充复用）
  - Surfaced by: 0A 前提 P3/P4（必要配套改动）
  - Files: realtime_serve/render_loop.py
  - Verify: pytest tests/test_t5_render_loop.py + 新序列号单测
- [ ] **T3 (P1, human: ~4h / CC: ~20min)** — unity_capture — 发送线程
  latest-wins→FIFO + preroll 门 + audio_start 回调 + underrun 统计
  - Surfaced by: S1/S4（触发点迁移 + 欠载策略）
  - Files: realtime_serve/unity_capture.py
  - Verify: pytest unity 相关测试 + 注入时钟 pacing 测试
- [ ] **T4 (P2, human: ~1h / CC: ~5min)** — unity_capture — 输出线程循环体
  try/except + 退避 + 错误计数（SPOF 防护）
  - Surfaced by: S1 SPOF 发现
  - Files: realtime_serve/unity_capture.py
  - Verify: 异常注入单测
- [ ] **T5 (P2, human: ~1h / CC: ~5min)** — main/config — --buffer-size/--preroll
  参数 + 范围校验 + timeBeginPeriod(1) + 配置回显
  - Surfaced by: S3（输入校验）+ 0E（delight 项）
  - Files: realtime_serve/main.py, config.py
  - Verify: 启动冒烟 --help
- [ ] **T6 (P2, human: ~1h / CC: ~5min)** — api — /status 扩展 6 字段（E2）
  - Surfaced by: S8 可观测性
  - Files: realtime_serve/api/app.py
  - Verify: GET /status 字段断言
- [ ] **T7 (P2, human: ~2h / CC: ~10min)** — render_loop — 漂移校正 E1
  - Surfaced by: 0D E1 采纳 + S4 长时运行边界
  - Files: realtime_serve/render_loop.py
  - Verify: 假时钟单测
- [ ] **T8 (P1, human: ~3h / CC: ~15min)** — tests — underrun→恢复集成测试
  + 敌意并发（put×flush×stop）
  - Surfaced by: S6 测试图
  - Files: tests/test_output_streamer_integration.py（新增）
  - Verify: pytest 集成组

## CEO 完成总结

```
+====================================================================+
|            MEGA PLAN REVIEW — COMPLETION SUMMARY (CEO)             |
+====================================================================+
| Mode selected        | SELECTIVE EXPANSION                          |
| System Audit         | 热路径连续8次perf提交,从严审查                |
| Step 0               | 方案B; E1/E2+delight并入,E3/E4→TODOS         |
| Section 1  (Arch)    | 3 issues（耦合回调/SPOF/回滚OK）              |
| Section 2  (Errors)  | 6 error paths mapped, 0 GAPS                 |
| Section 3  (Security)| 0 issues                                     |
| Section 4  (Data/UX) | 7 edge cases mapped, 0 unhandled             |
| Section 5  (Quality) | 0 issues                                     |
| Section 6  (Tests)   | 图已产出, 0 gaps（时钟注入防flake）           |
| Section 7  (Perf)    | 0 issues（550MB已文档化）                     |
| Section 8  (Observ)  | 6 新字段/4 日志点补齐                         |
| Section 9  (Deploy)  | 0 risks（revert即回滚）                       |
| Section 10 (Future)  | Reversibility: 4/5, debt items: 1(减债)      |
| Section 11 (Design)  | SKIPPED（无UI范围）                           |
+--------------------------------------------------------------------+
| NOT in scope         | written (5 items)                            |
| What already exists  | written                                      |
| Dream state delta    | written                                      |
| Error/rescue registry| 6 methods, 0 CRITICAL GAPS                   |
| Failure modes        | 6 total, 0 CRITICAL GAPS                     |
| TODOS.md updates     | 2 items proposed (E3/E4)                     |
| Scope proposals      | 6 proposed, 4 accepted (SEL)                 |
| CEO plan             | 本文件即计划                                  |
| Outside voice        | subagent-only（Codex未安装）                  |
| Lake Score           | 4/4 选项选了完整方案                          |
| Diagrams produced    | 3（架构/数据流/测试图）                       |
| Stale diagrams found | 0                                            |
| Unresolved decisions | 0（ taste 项留最终门）                        |
+====================================================================+
```

**UNRESOLVED DECISIONS:**（无——taste 项移交最终门：buffer-size 默认 50 的
内存/延迟权衡、preroll 欠载重武装策略）

## CEO 双声部（Step 0.5）

CODEX 不可用 → [subagent-only]。

### CLAUDE SUBAGENT（CEO — 战略独立挑战）核心发现

1. **[critical] 根因三件套被划出范围是本末倒置**：`PERFORMANCE_REPORT.md`
   已定位三个尖峰源（face_repair ONNX p95=16ms 周期尖峰、Generator p95
   40-84ms、batch 边界 ~19fps 下探=GPU→CPU 与 compositor 互等锁）——
   正是 18-25 波动中"18"的来源。三者均有以天计的修法，修掉后均值方差同降，
   现有 sender 精确 pacing 输出即平滑。缓冲池只是把吸收深度从 1 帧扩到 50 帧。
2. **[high] timeBeginPeriod(1) 应先单独实验**：~5 行进程级修复，加上后现有
   render loop pacing 同样变准，可能消掉大半"pacing 抖动"论据。
3. **[high] 余量未推演**：30+fps 观测大概率来自 idle 模式（无 lip chunk、
   无 face_repair 尖峰）；语音模式净富余可能仅 5 帧/s，50 帧回填需 10s，
   一次 500ms 停顿吃 12 帧。默认 50 无数据支撑。
4. **[critical] DAC 延迟补偿丢失**：现行 `render_loop.py:328` 的
   `elapsed - _output_latency()` 在 seq 重写中被删。pop 时 push PCM，PCM
   再走 30-100ms 输出延迟才出声 → 口型系统性领先音频。
5. **[critical] 欠载/漂移校正/re-arm 三机制互打架**：语音中欠载 → 音频继续
   走而 seq 停 → 永久失步；漂移校正 2s 内必触发 → 已烘焙帧口型跳变；
   re-arm preroll = 冻结 2s。需明确协议：语音中只重发不 re-arm、漂移校正
   仅 idle 期生效、base_seq 平移只影响未生产帧。
6. **[high] flush/段切换并发协议缺失**：/segment/switch 不 flush → 旧段画面
   继续上屏最长 2s（现状 ≤1 帧），Unity 侧体验回归。
7. **[high] 段首欠载矛盾**：flush 后近空缓冲 + 首块 lip 100-200ms 占用 →
   输出线程必欠载，与"段间低延迟"自相矛盾。需段首小阈值（~5 帧）快速预填充。
8. **[medium] 内存总账**：preloader ~1.1GB + 缓冲 528MB + 拷贝 ≈ 2GB；
   缓冲应环形预分配，flush 瞬间 11MB×50 再分配抖动。
9. **[medium] 被草率排除的中间档**：深度 4-8 的小 FIFO（~320ms 吸收 p99 尖峰）
   不需要 seq 重写、不需要 preroll、改动收敛在 unity_capture.py 一处。

### CEO DUAL VOICES — CONSENSUS TABLE
```
═══════════════════════════════════════════════════════════════
  Dimension                           Claude  Codex  Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Premises valid?                   部分     N/A    DISAGREE→门
  2. Right problem to solve?           部分     N/A    DISAGREE→门
  3. Scope calibration correct?        否       N/A    DISAGREE→门
  4. Alternatives sufficiently explored? 否     N/A    DISAGREE→门
  5. Competitive/market risks covered? N/A     N/A    N/A
  6. 6-month trajectory sound?         有条件   N/A    N/A
═══════════════════════════════════════════════════════════════
```
Codex 未安装 → 单声部。声部与主审在"顺序"上分歧（主审选方案 B 直接做；
声部要求先修根因+实测再定深度）→ **USER CHALLENGE，移交最终门**。
声部的技术缺陷发现（4/5/6/7/8）为主审确认成立 → 直接修正方案，无需用户裁决。

## 方案修正（声部确认的技术缺陷，任何缓冲变体都需要）

| # | 修正 | 内容 |
|---|------|------|
| F1 | DAC 补偿保留 | audio_start 标记前移 `round(output_latency_s × fps)` 帧；漂移校正基线含固定偏移 |
| F2 | 欠载协议 | 语音激活期：只重发上帧、不 re-arm、不漂移校正；漂移校正仅 idle 期执行；base_seq 平移只影响未生产帧，单次步长 ≤2 帧 |
| F3 | flush 状态机 | idle→audio 迁移原子化：flush 与首帧产出互斥（复用 _push_lock + buffer 锁序文档化）；/segment/switch 同协议 flush；验收用例覆盖"语音中 push_full"与"缓冲满时切段" |
| F4 | 段首快速预填充 | 段首 preroll_ss=min(5, total_lip_frames) 与启动 preroll 区分；验收指标=推音频→首帧口型上屏端到端延迟 |
| F5 | 内存总账+环形预分配 | 列全进程预算（preloader 1.1GB + 缓冲 + 拷贝）；缓冲环形槽位预分配，杜绝 flush 抖动 |
| F6 | timeEndPeriod 配对 | timeBeginPeriod(1)/timeEndPeriod(1) 成对，退出清理 |

## 新增任务

- [ ] **T0 (P1, human: ~2h / CC: ~15min)** — 测量 — timeBeginPeriod(1) 单独
  实验 + 语音模式生产耗时分布采集（p50/p95/p99/最长停顿，≥10min，face_repair
  开、摄像头开），落盘测量脚本与报告
  - Surfaced by: CEO 声部发现 1/2/3/4
  - Files: realtime_serve/main.py, 新增 measure 脚本
  - Verify: 报告文件存在且含分布数据
- T1-T8 相应调整：缓冲深度与是否 seq 重写由 T0 数据决定（见最终门选项）。

## 修订后前提表

| 前提 | 修订 |
|------|------|
| P1 生产能力 | **需 T0 实测**：30+fps 为 idle 模式观测，语音模式净富余未知 |
| P2 缓存可稳 25fps | 不变，但深度由 T0 数据决定（可能 ≤8 帧） |
| P5 内存 | 修正为全进程 ~2GB 总账，环形预分配 |

---

# DX 审查（Phase 2.5）— 模式: DX POLISH

产品类型：本机运营的推理服务（操作者=开发者本人）。Persona：solo operator，
通过 CLI 启动、经虚拟摄像头供 Unity 消费。

## 开发者旅程图

```
发现→评估→安装→hello world→集成→调试→升级→扩展→迁移
 ✓    ✓    ✓   ~2min(模型加载主导) ✓   ⚠新增   ⚠语义变更  N/A  N/A
```

## 8 维评分（0-10，POLISH 模式：逐项修至达标）

| 维度 | 分 | 证据与差距 | 修复 |
|------|----|-----------|------|
| 1 Getting started | 7 | TTHW ~2-3min（模型加载为地板，非本方案范围）；新增 2s preroll 启动延迟是无预告的意外 | T5：启动 banner 回显配置+延迟/内存代价 |
| 2 CLI ergonomics | 8 | --buffer-size/--preroll 可猜；但 **--target-fps 语义变更不可发现**——老习惯命令行为不同 | T5：--help 文案改为"限制最终向摄像头输出的帧率"；banner 说明语义变更 |
| 3 Error messages | 5 | 现状 "[AudioPlayer] Underflow" 无原因无对策；新欠载日志若只计数同样不合格 | T3/T6：欠载日志规格=问题+原因+对策（"production Xfps < output Yfps 持续 Zs；建议 --no-face-repair 或降 --target-fps"） |
| 4 Docs | 6 | --help/启动横幅需更新；持续 underrun 无 runbook | T5 扩围：README/PERFORMANCE_REPORT 增补欠载处置段落 |
| 5 Escape hatches | 9 | --buffer-size 1（近旧行为）、--preroll、--no-face-repair 均可覆盖默认 | 达标 |
| 6 Credible | 8 | 行为可预测；preroll 期间 /status 显示 pre-rolling 防黑盒 | T6 已含 |
| 7 Findable | 7 | /status 字段自说明；underrun 计数可查 | 达标 |
| 8 Valuable | 9 | 输出稳定即核心价值 | 达标 |

**DX Scorecard 总分: 7.4/10 → 修复后 8.9/10**（T3/T5/T6 落地后）
**TTHW: ~2-3min → 目标 <2min**（模型加载为地板，本方案不触碰；preroll 仅 +2s）

## DX 新增/修订任务

- [ ] **T9 (P2, CC: ~5min)** — 欠载/preroll/flush 日志文案规格（问题+原因+对策
  三段式），写入 T3 实现规格
  - Surfaced by: DX 维度 3（5/10）
  - Files: realtime_serve/unity_capture.py, api/app.py
  - Verify: 人为制造欠载，检查日志含对策建议
- T5 扩围：--help 文案、启动 banner（语义变更+内存+延迟回显）、README runbook。

## DX 双声部（Step 0.5）

CODEX 不可用 → [subagent-only]。

### CLAUDE SUBAGENT（DX — 独立体验挑战）核心发现

- **[高] --no-unitycapture 下音频全静默**：音频触发点在发送线程，
  `unity_capture=None` 时 push_pcm 永不触发——日常测试入口直接坏。→ 修复
  F7/T9：输出消费者与摄像头解耦，无摄像头时起无画面消费者线程。
- **[高] preroll > buffer-size = 静默永久卡死**：启动校验缺失。→ F8。
- **[高] --target-fps 语义翻转零文档任务**：--help/模块 docstring/config 注释/
  PERFORMANCE_REPORT 验证段等 7 处旧描述。→ T5 扩围已列 7 处清单。
- **[高] 欠载重武装默认组合 = 体验倒退**：每次 push_full flush 清空后以净速率
  ~5帧/s 回填，20-40s 脆弱窗口内一次 40ms gen 抖动 → re-arm preroll=50 →
  冻结 2s，比要修复的单帧重复更糟。→ F9（重武装阈值解耦 min(preroll,5)、
  flush 保留小 reserve）。
- [中] preroll 2s 静默期无进度行；flag 无单位且与 max_buffer_seconds 单位相反；
  buffer 持续满不可观测；RUNBOOK 无落点（指定 realtime_serve/RUNBOOK.md）；
  /status 旧 fps 字段语义未定；内存/延迟代价不进运行时回显。
- [低] buffer-size 无上界（手滑 5000 → 55GB）；timeBeginPeriod 失败静默；
  "启动即推音频跳过 preroll"捷径未文档化。

### DX DUAL VOICES — CONSENSUS TABLE
```
═══════════════════════════════════════════════════════════════
  Dimension                           Claude  Codex  Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Getting started < 5 min?          是+1缺口 N/A    部分确认
  2. API/CLI naming guessable?         否(单位) N/A    DISAGREE→修正
  3. Error messages actionable?        否       N/A    DISAGREE→修正
  4. Docs findable & complete?         否       N/A    DISAGREE→修正
  5. Upgrade path safe?                否(旁路) N/A    DISAGREE→声明
  6. Dev environment friction-free?    是       N/A    CONFIRMED
═══════════════════════════════════════════════════════════════
```

## DX 修正（并入方案）

| # | 修正 | 内容 |
|---|------|------|
| F7 | 消费者与摄像头解耦 | OutputStreamer 独立于 unity_capture；--no-unitycapture 时照样运行（仅无 send），音频路径与 pacing 保持；集成测试补"无摄像头推音频有声" |
| F8 | 参数校验 | preroll ≤ buffer-size 否则报错退出；buffer-size 上界 300；单位写入 help（帧、≈N/25 秒、~11MB/帧） |
| F9 | 欠载重武装解耦 | 重武装阈值 = min(preroll, 5) 帧固定默认；flush 保留 5 帧 reserve 而非清空；underrun 日志附双 fps 对比与对策 |
| F10 | 可观测补全 | preroll 进度行 [preroll] N/M；buffer 持续满 >Ns 告警（输出侧停止消费→查 Unity/OBS 捕获端）；timeBeginPeriod 失败打警告 |
| F11 | 文档落点 | 新建 realtime_serve/RUNBOOK.md（欠载/缓冲满处置）；/status 保留旧 fps=output_fps 一版并标注弃用；字段名统一 underruns |

任务清单合并：T5（参数+文档 7 处）、T6（/status）、T9（解耦+无摄像头音频测试）
为一级任务。

---

# 工程审查（Phase 3）— 最终门前的收官审查

## Step 0 范围挑战（autoplan: 不缩减）

- 复杂度检查：7 文件 + 2 新组件 → 低于 8 文件阈值。通过。
- 子问题→现有代码映射：见 0B。无重复建设。
- CEO 声部的顺序挑战（根因优先 vs 缓冲优先）→ 已作为 USER CHALLENGE
  移交最终门，此处不缩减也不裁决。
- 结论：范围成立，**架构修正 E1 如下**。

## S1 架构审查（依赖图 + 单时钟归属）

**发现 E1 [high]：时钟归属必须唯一，OutputStreamer 独立成类。**
现状 UnityCaptureSender 内含 pacing 线程（latest-wins）。若在其上再叠 FIFO，
会出现文档自述警告过的"两个自由运转的 40ms 时钟互拍"。且"发送摄像头的类"
拥有"音频起点触发"职责错位。

裁决（P5 显式优于聪明）：
```
改造后依赖图:
                                              ┌──────────────────┐
 API ─▶ RenderLoop(生产,不限速) ─put──▶ FrameBuffer ─get─▶ OutputStreamer ─send─▶ UnityCaptureSender
 │                                        │(FIFO+preroll)   │ 唯一 pacing 时钟      │ (退化为同步
 ▼                                        │                  │ ├─ 段首帧→audio回调    │  sink, 无线程)
 AudioPlayer ◀────push_pcm────────────────────────────────────┘ └─ 空缓冲→重发+计数  └──────────┘
```
- UnityCaptureSender：移除内部线程与 latest-wins（start/submit 废弃），
  保留构造/`_send_direct`/close。进程内**唯一** pacing 时钟在 OutputStreamer。
- RenderLoop 不再 import unity 相关；生产者与摄像头彻底解耦。
- 音频回调由 main.py 注入 OutputStreamer（依赖倒置，发送层不知音频存在）。

## S2 代码质量

DRY：pacing 循环现存两份（render_loop._run、unity sender loop）→ 收敛为
OutputStreamer 一份。命名：FrameBuffer/OutputStreamer 直述职责。无圈复杂度
超标。无发现（除 E1）。

## S3 测试审查

测试图与防 flake 策略见测试计划产物
（~/.gstack/projects/Eikwang-EDTALK/edtalkdev-main-test-plan-20260902.md）。
追加 E1 影响：unity 测试改造范围扩大（移除线程测试，新增 OutputStreamer
pacing/underrun/回调测试）。T8 相应更新。无未覆盖路径。

## S4 性能

GIL 争用：OutputStreamer 的 send（11MB 拷贝 ~5-10ms）持 GIL 会短暂拖停
生产者 → 填池期由缓冲吸收，稳态期生产者本就阻塞在 put 上，无影响。
内存总账见 F5。无 N+1 类问题。无新发现。

## 工程完成总结

```
+====================================================================+
|         MEGA PLAN REVIEW — COMPLETION SUMMARY (ENG)                |
+====================================================================+
| Step 0 范围挑战      | 范围成立; 架构修正 E1(单时钟归属)             |
| Section 1 (Arch)    | 1 issue (E1) — 已裁决采纳                     |
| Section 2 (Quality) | 0 issues (DRY 收敛为 E1 一部分)               |
| Section 3 (Tests)   | 测试图+产物落盘; E1 扩围已并入 T8             |
| Section 4 (Perf)    | 0 issues (GIL 争用被缓冲吸收)                 |
| NOT in scope        | 已写（5 项）                                  |
| What already exists | 已写                                          |
| Failure modes       | 6 项, 0 CRITICAL GAPS                         |
| Test plan artifact  | 已写盘 ✓                                      |
| TODOS 收集          | E3 自适应preroll, E4 帧直方图, 根因三件套(若门裁缓冲优先), 批量推理 |
| Unresolved          | 0（顺序问题留门）                              |
+====================================================================+
```
