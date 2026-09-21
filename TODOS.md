# TODOS.md

> 由 /autoplan（2026-09-03）自动写入——收集 CEO/DX/ENG 三阶段的 Deferred 项。
> 每项含动机与上下文，3 个月后接手也能看懂为什么。

## P2

### 音频感知的段边界切换（播完当前音频再切）
- **What:** 边界切换的触发条件从"按片段循环周期"（视频墙钟）扩展为"音频播放条件判断"：若当前有音频正在合成（Audio2Lip 异步推理未完成）或正在播放（唇形系数仍在消费），则顺延切换；等音频合成完毕且播放完毕后再切。
- **Why:** 直播场景中 TTS 音频进行到一半时切段，音频会继续播但画面内容跳变（嘴型突换新段）。用户真机验收（2026-09-03）确认需要此优化——"边界切换需要优化，将按片段循环周期改为按音频播放条件判断"。
- **Pros:** 切段永不打断"正在说的话"；与 AI-Vtuber 反应段联动（弹幕触发切段）体验正确。
- **Cons:** 边界判定复杂化（读渲染循环的合成/播放状态）；无音频时回退循环周期语义；连续 TTS 轰炸可能无限顺延——需要顺延上限策略（如最多 N 秒强制切）。
- **Context:** 当前实现：`RenderLoop._check_segment_schedule` 用视频墙钟判定边界（`elapsed*fps >= total`），调度器 advance 与 pending segment_end 都在此触发（render_loop.py）。音频状态已在 RenderLoop 现成可用：`_lip_filled/_total_lip_frames`（合成进度，合成中 = filled < total）、`_audio_start_time`（播放锚点，`AUDIO_END_GRACE_S=1.0` 播完宽限）。判定建议：`合成中 or 播放中 → 顺延`。落地时需同步更新 EMBEDDING.md 的 `at=segment_end` 语义描述与 RUNBOOK。
- **Effort:** M（human ~1-2天 / CC ~1h）
- **Priority:** P2
- **Depends on:** 音频播放功能优化（用户规划中，先行）。

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

## P2

### 一键角色适配管线（微调自动化编排）
- **What:** 采集→预处理→微调→自动评估→评分门限→热切换 的单命令编排脚本。
- **Why:** 微调修复计划（FINE_TUNE_FIX_PLAN.md）落地后，每个新角色的适配仍是 4-5 步手工流程；编排化后"加一个角色"的成本从小时级降到一键。
- **Pros:** 角色库规模化前提；评分门限自动判定避免劣化模型上线。
- **Cons:** 依赖训练线与评估协议（D）稳定后再封装；编排层有跨环境（训练 venv/推理 runtime312）复杂度。
- **Context:** /autoplan（2026-09-04）CEO 阶段 DEFER 项 — 12 个月理想态的核心能力，但当前优先修复微调本身。
- **Depends on:** FINE_TUNE_FIX_PLAN 的 T3 评估协议 + 训练线验收达标。

## P3

### 多角色微调回归套件
- **What:** 每个已适配角色的 eval 指标快照入库，新训练代码改动后批量回归。
- **Why:** 微调代码演进（损失/采样改动）可能对已交付角色产生未察觉的退化。
- **Pros:** 角色库质量有人守；训练改动敢做。
- **Cons:** 需要角色库积累后才有价值；跑批耗时。
- **Context:** /autoplan（2026-09-04）CEO 阶段 DEFER 项；评估协议 eval_finetune.py 已含 JSON 归档，本项是其多角色扩展。
- **Depends on:** 一键角色适配管线；≥2 个已适配角色。

## P3

### patch 判别器嘴区 GAN
- **What:** 嘴区局部 patch 判别器，实现 GAN 级别的嘴区对抗加权。
- **Why:** 嘴区加权损失（H）只能作用于 vgg/l1 — 全图 Discriminator 结构上无法按区域加权（ENG 发现 E2）；patch 判别器可补齐对抗信号的区域性。
- **Pros:** 嘴部纹理对抗信号更集中。
- **Cons:** 新增判别器 = 新基础设施 + GAN 平衡重调；H 的 vgg/l1 加权可能已足够。
- **Context:** /autoplan（2026-09-04）ENG 阶段拒绝项留档 — 先看 H 的效果再决定。
- **Depends on:** H（嘴区加权）实验结果不足时再启动。

## 视线纠正时机优化——延迟项（2026-09-21 /autoplan）
- [F] Generator 眼部忠实度微调（根因层解法）：当前眼像素经贴回补丁获得；微调使 Generator 重绘本身眼部忠实后可撤贴回。依赖：微调管线（fine_tune/ 既有）。优先级 P2。Context: GAZE_TIMING_PLAN.md §4-F。
- [X3] 全脸身份保真审计：逐区域 probe orifra vs 生产域输出，量化"不像原图"的非眼区域贡献。优先级 P2。Context: CEO 声部发现1。
- [战略] 生成时视线/眼部条件化：商业级数字人竞品的结构性路线（无贴回接缝/闪烁类伪影）。优先级 P3。Context: CEO 声部发现5。
- [perf] pristine ori_crop_256 GPU 批量预载：省每帧 0.3ms H2D。当前非瓶颈。优先级 P3。
- [既有] EAR 阈值 256 域校准；gaze off→on 运行时懒初始化（/gaze/params 409 现状）。继承自 GAZE_CORRECTION_PLAN 遗留。
