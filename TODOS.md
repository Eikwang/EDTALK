# TODOS.md

> 由 /autoplan（2026-09-03）自动写入——收集 CEO/DX/ENG 三阶段的 Deferred 项。
> 每项含动机与上下文，3 个月后接手也能看懂为什么。

## P2

### gaze off→on 懒初始化（corrector 懒构造）
- **What:** 运行时从 gaze off 热启校正：懒构造 GazeCorrector（载入 ckpt + try_capture_graph）+ set_eyes_enabled(True) + 双槽掩码重算。
- **Why:** 默认 `--gaze-correction off` 启动后无法运行时开启（主计划 M2 定为 409 拒绝）。直播中"临时想开眼校"需重启。
- **Pros:** 去掉最常见的重启场景之一；409 语义可退役。
- **Cons:** 请求线程秒级 GPU 操作需异步化（不能进事件循环/渲染线程）；双槽 mask 一致性要靠 mask epoch。
- **Context:** 三声部一致确认 off→on 不是"只差暴露"（corrector 未构造 + 眼区 mask 联动，main.py:309-322 / sequence_frame_pack.py:232）。入口已留好：`POST /gaze/params` 的 409 分支与 `gaze_capable` 能力位。
- **Depends on:** 控制面 facade 落地（本计划）；mask epoch 机制（M18）。

## P2

### flush keep_reserve 平滑保留策略（immediate 档进一步消冻结）
- **What:** immediate 切段保留部分旧段尾帧 + 预填 ≥4 帧新段帧后解除 preroll，替代全弃。
- **Why:** M8 让 segment_end 达到帧级无缝，但 immediate 档仍有 ~0.3s flush+preroll 不连续（诚实承诺保留）。
- **Pros:** 前端"立刻换人设"按钮的观感提升。
- **Cons:** 帧段归属判定复杂；收益场景少于 segment_end。
- **Context:** render_loop.py `_flush_buffer(keep_reserve=0)`（E2 决策）与 `reset_preroll` 是冻结来源；`keep_reserve` 参数路径已存在。
- **Depends on:** 本计划切换语义（M3/M8/M15）稳定后。

## P3

### 参数持久化（API 设置重启保留）
- **What:** /gaze/params 等运行时设置的持久化（JSON 侧车文件或嵌入方自管）。
- **Why:** 现在运行时值覆盖 CLI 但仅存活至重启（M2 优先级规则）。
- **Pros:** 直播机重启后免手动恢复。
- **Cons:** 覆盖策略要先定（覆盖 CLI 还是叠加）；嵌入方可能更想自己管配置。
- **Context:** 持久化是嵌入方常见自管项——先问 AI-Vtuber 走查结论再动（契约 v0 已登记缺口）。
- **Depends on:** AI-Vtuber 对接 spike（主计划 T1）。

## P3

### WebSocket/事件推送状态变化
- **What:** 段切换/参数变更/预取状态的事件推送通道。
- **Why:** 嵌入方前端目前只能轮询 /status。
- **Pros:** 前端实时联动，免轮询。
- **Cons:** 协议未定型；facade 就位后是加法。
- **Context:** 三阶段一致 Deferred——嵌入方协议定型后再做；落点 = facade 加 subscribe() 方法。
- **Depends on:** AI-Vtuber 对接 spike。

## P3

### 多角色（character）运行时切换
- **What:** 运行时加载另一 character_dir（换数据包 + 权重路径），facade 增 `load_character()`。
- **Why:** 直播"换皮"是高频操作；CharacterDataPack 绑定单 character_dir（sequence_frame_pack.py:444）。
- **Pros:** 消灭最大已登记契约缺口。
- **Cons:** 双 pack 内存策略（~2-3GB/角色）需设计；切换为秒级停顿，前端不得按无缝预期设计。
- **Context:** 契约 v0 已登记缺口（M14）；先做 spike 确认真实需求频度。
- **Depends on:** AI-Vtuber 对接 spike。

## P3

### CHANGELOG.md 与版本化策略
- **What:** 建立 CHANGELOG 与 facade 方法面版本记录（v0 → 1.0 冻结流程）。
- **Why:** DX Pass 5 升级路径 6/10；嵌入方升级子模块无回归网说明。
- **Pros:** 升级不可怕；破坏性变更有登记处。
- **Cons:** 维护成本（低）。
- **Context:** 契约冻结规则（M14）已写明破坏性变更须登记——CHANGELOG 是登记处。
- **Depends on:** 无。

## P3

### CI 流水线（pytest + 统一环境冒烟）
- **What:** GitHub Actions：runtime 环境跑 tests/ + 统一环境 `import realtime_serve.service` 冒烟。
- **Why:** ENG 验收（M13）要求统一环境 import 冒烟可重复执行；当前全靠手工。
- **Pros:** 嵌入可行性回归有人守。
- **Cons:** GPU 测试在 CI 上不可行——冒烟仅 CPU 侧 import 检查。
- **Context:** pyproject 依赖为空是现实约束；constraints.txt 已存在于 D:\AI\install。
- **Depends on:** 统一环境工程（AI-Vtuber spec）稳定。
