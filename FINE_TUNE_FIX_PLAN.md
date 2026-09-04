<!-- /autoplan restore point: /c/Users/admin/.gstack/projects/Eikwang-EDTALK/main-autoplan-restore-20260904-092651.md -->
# EDTalk 微调训练修复计划（FINE_TUNE_FIX_PLAN）

> /autoplan 产出（CEO + Eng 双阶段评审后修订版 v2）。目标：修复 `train_fine_tune.py` 微调流程，在保持底模 pose/mouth 解耦能力的前提下提高单人生成质量。

## 1. 问题陈述与产品消费面

对底模 `EDTalk_lip_pose.pt` 用单角色数据集微调后出现三个症状：
1. 推理训练帧序列时角色头部姿势变化，无法像底模那样保持原图姿势只改口型；
2. 背景变形，头发、衣服模糊扭曲；
3. 实时推理将嘴部贴回原图后嘴巴变大、位置不准。

**产品消费面修正（CEO 评审）**：实时链路 compositor 只把嘴区贴回原图（compositor.py:167-183，mask 外像素全部丢弃）。因此症状 2 在最终直播输出中不可见 — 但它是模型劣化的重要**诊断信号**（反映 direction 膨胀，同样污染嘴区），保留为评估指标而非产品目标。产品真正消费的是：**嘴区渲染质量、嘴区与原图几何一致性、mask 对齐**。

已试无效：`--only_fine_tune_dec`（只训 dec）、全量训练、`--freeze_direction`。

## 2. 调查结论：根因分析（代码证据链，独立声部已逐条验证）

### 2.1 实时推理为何能保持原图姿势

- `realtime_serve/render_loop.py:727`：`infer_frame(cropped, alpha_D_lip, cropped)` — **pose 驱动帧就是 source 帧自身**。
- `realtime_serve/models.py` → `Generator.test_from_audio_pose_image`（networks/generator_lip_pose.py:181）：pose alpha = `enc(source)→fc→pose_fc`，lip alpha = Audio2Lip 音频输出（20 维）。
- `realtime_serve/compositor.py:219-225`：mask 定位基于**原图帧**的 68 点 facemark。
- 结论：底模下 dec 对 pose alpha 序列（source 视频自身的头动，幅度小）的响应正确，输出头位 ≈ source 头位，落在贴回容差内。**渲染姿势一漂移，固定 mask 贴回必然错位放大（"嘴变大"的直接机制）。**

### 2.2 底模如何建立解耦（step2，trainer_Mouth_Pose_decouple.py）

- 六元组输入：`img_a_identity`（身份帧）、`img_a`（驱动帧）、`imagea_b`（**landmark 仿射嘴部移植合成帧**，dataset_HDTF_lip_nonlip.py `get_IA_MB`）。
- 跨身份重建 `recon_a_cross = dec(wa_a_identity + dir(lip_B, pose_A), feats_a_identity)`。
- **显式解耦监督**：`lip_space`/`pose_space` 余弦一致性损失（trainer_Mouth_Pose_decouple.py:174-177）— **梯度流向 enc/fc/direction**（损失定义在 direction 潜码上），即底模中它训练的是"编码器学会 移植嘴=真嘴 / 合成姿势=真姿势"。
- GAN 仅监督 cross 重建。lr=0.002, batch=4。

### 2.3 微调链路的缺陷（根因，按影响排序；机制经独立声部修正）

**R1（主根因）解耦监督缺失 + 训练/推理 pose 条件错位**
- 微调只有单一重建目标 `dec(wa_s + dir(target), feats_s) → target`（vgg+l1+gan+0.5·自重建）。step2 的 lip_space/pose_space、跨身份重建全部不存在。
- **精确机制（修正版）**：训练中 pose alpha 永远来自 target 帧（`wa_t→fc→pose_fc`），dec 被迫把"latent pose 分量 → 输出 target 头位"重写为绑定映射；推理时 pose alpha 来自 **source 自身** — 这个恒等条件在随机两帧采样下从未被显式训练，dec 重新校准后落点漂移。同身份随机两帧下 feats_s 的头位与 target 高度相关，进一步允许 dec 弱化 pose direction 依赖、从 skip 特征近似头位。
- **lip→头泄漏**：单人数据中说话伴随轻微头动 → lip direction 学会带动头部运动 → 推理时每帧音频不同 → 头位逐帧漂移。
- **direction 膨胀**：direction 被迫解释帧间全部差异（亚像素背景移动、压缩噪声）→ 推理时小 alpha 落在 dec 未校准区 → 背景/头发被重绘扭曲。

**R2 学习率与冻结策略失当**
- `--lr` 默认 0.001（帮助文本却写 0.0001，矛盾），对微调过高。
- 全量模式 `direction_lr_ratio=5.0` 快速改写 Direction/fc — Audio2Lip 输出只在底模 lip 空间有效，一动就接口失配。
- `orthogonality_loss`（0.01）是底模没有的新正则。

**R3 Discriminator 缺陷叠加**
- 底模 dis 的 minibatch stddev 用 per-sample `repeat_interleave`，微调 dis 是标准 per-group `repeat` — 权重可加载但统计口径不同。
- 单人小数据 D 快速过拟合 → GAN 梯度劣化高频细节。**当前方案对 R3 无任何动作**（见 C 的 `--gan_weight` 消融开关）。

**已排除**：网络结构不匹配 — ckpt 严格加载验证通过（gen strict=True 成功；dis 50 张量零 missing/unexpected；direction weight 512×26）。

### 2.4 症状映射

| 症状 | 根因 | 产品可见性 |
|---|---|---|
| 姿势无法保持原图 | R1（pose 条件错位 + lip→头泄漏）+ R2 | 直播输出可见（贴回错位） |
| 背景变形/发丝模糊 | R1（direction 膨胀）+ R3 | 诊断信号（直播输出不可见） |
| 贴回后嘴大/位置不准 | 姿势漂移 × 固定 mask 错位放大 | 直播输出可见 |

## 3. 修复方案（v2：优先级经独立声部挑战后重排）

**核心策略**：先用便宜杠杆（C+H）止血、用 D 建立可信基线、B 对齐训练/推理分布；A（cross_recon 锚定）只在 B+C+H 不足时启动；G（推理侧对齐）作为独立缓解层并行 — **不把全部赌注押在训练侧**。

### C（P0 止血，10 分钟）：参数修正
- `--lr` 默认改 0.0001，帮助文本对齐。
- **`--batch_size` 默认 8→4（ENG 声部最终实测）**：现状 batch8 含 VGGLoss 实测 **15.05GB/16GB + 2535ms/步**（无 VGG 时 782ms — 3.2 倍阶跃是 WDDM 溢出共享内存的指纹，很可能是历史崩溃的未记载诱因）；batch4 含 VGG 12.03GB + 532ms — **每样本反而快 ~2.4 倍**，"降 batch 训练变慢"不成立。batch6 雪崩（15.44GB/4396ms）禁区。
- `direction_lr_ratio` 默认 1.0；`orthogonality_loss` 权重默认 0。
- **新增 `--gan_weight`（默认 1.0）**：GAN 损失消融开关 — 单人小数据 D 过拟合时置 0 验证 R3 贡献（回应独立声部"R3 无动作"缺口）。

### H（P0，与 C 同批）：嘴区加权损失（v3 修正）
训练目标对齐产品消费面。**权重设计（ENG 声部 H2 修正）**：嘴区 mask（68 点 facemark 凸包外扩）内 ×4，脸部其余 ×1，**背景 ×1.0（不是 0.5）** — 背景锚定是压制 direction 膨胀的约束，降权会亲手踩爆自己的背景 SSIM 诊断门槛；"背景 ×0.5"降为消融可选项。
**GAN 不加权（写死，ENG 声部 H1）**：Discriminator 是全图单 logit 判别器，无空间维度可加权；嘴区裁剪送 D 会破坏底模 dis 权重感受野（等于重训 D），patch-D 是新基础设施 — 唯一干净答案是 GAN 保持全图，H 只作用于 vgg/l1/self_recon。
**VGG 实现细节**：特征金字塔各尺度 `F.interpolate(mask)` 加权后**按该尺度权重均值归一化** — 否则 ×4/×1 改变总损失量级，等价于顺手动了 lr，污染 C 项消融对照。
**mask 数据耦合（ENG 声部 A2）**：mask 缓存落 `<datapath>/mouth_mask/`（schema version）；crop_head_video 已存 `<dataset>/facemark/<video>.npy`（256 空间，NaN 行=检测失败）与帧序的耦合是隐式的 — dataset 读 mask 时校验 `len(mask)==len(frames)` 且帧 stem 连续，不一致 fail-fast；NaN 行 → 回退全 1 权重 + 计数日志（不炸训练）。禁止在 DataLoader worker 里跑 dlib。

### B（P0 主项）：采样策略对齐推理分布（v3 修正）
- **时序近邻帧对**：target = source 邻域 ±K（随机 1..64），姿势差小、口型差大 — 正是推理分布。
- **混入 5-10% 同帧自对（source==target）**：显式训练推理时的"pose alpha 来自自身"恒等条件（独立声部指出的最高杠杆点）。
- **远对 ≥30%（v3，ENG 声部 E3）**：近邻对会收缩训练 lip alpha 方差，与推理时 Audio2Lip 全量程输出脱节；远对保底 30%，可选混入少量"lip alpha 直用 Audio2Lip 输出（配 mel）"样本（仓库已有 mel/lip_feature 管线）。
- **held-out 排除（ENG 声部 E1，high）**：held-out 帧索引必须从**所有采样路径**（近邻/自对/远对/移植对）剔除 — 否则邻域窗口直接采到评估帧当 target，全部指标静默虚高、决策门失效。单测断言"held-out 索引永不出现在训练 batch"。
- **近邻限定同视频内**（多场次单角色时跨"视频"=跨场次，近邻无意义）；移植对口型差阈值用内唇 60-67 点距离实测标定。
- 单帧回退与越界 clamp 保留（参考 `obtain_seq_index` 模式）。

### D（P0，先于 A）：评估协议 `tools/eval_finetune.py`
- **先跑当前三个失败 ckpt 的基线数字** — 验收阈值从实测得出，不拍脑袋。
- **held-out 帧集**：视频末 10% 帧不参与训练，专供评估（当前 dataset 无 val split，训练帧指标会掩盖过拟合）。
- 四组指标：姿势保持 MAE（landmark euler/像素）；lip 保真（同 Audio2Lip alpha 过底模/微调模型，嘴区像素差与开合幅度比）；背景 SSIM/PSNR（**诊断指标**，非产品目标）；E2E 贴回错位像素。
- **自校准基线**：底模 vs 底模输出应零差异，校准指标上界（注意：纯模型指标此项是恒真式，只有 E2E 贴回这类含 landmark/mask 管线的指标才有非平凡噪声底 — 表述区分）。
- **eval 元测试（ENG 声部 T1，high）**：自校准只覆盖"无差异"方向，还需 (a) 已知扰动敏感性 — 嘴区平移 3px/模糊/姿势轻推，断言指标按预期方向量级响应；(b) 确定性 — 同一 eval 跑两遍 JSON 逐位相同（compositor 的 `_smoothed_facemark` EMA 是有状态，跨视频复用会污染指标，必须每次重置）；(c) NaN/检测失败路径。
- 对比组天然包含"不微调的底模"（Alt-4 锚点）。
- **权重插值回退扫描（v3，ENG 声部 H4）**：direction 的 QR 分解使全量插值非线性（Q 列符号/排序翻转 → λ 曲线非单调）。分组插值：`{dec,enc}` 插值 + direction/fc 系保持底模（首选）；全量插值仅作对照。λ∈{0,0.25,0.5,0.75,1.0}。

### A（P1，条件触发：B+C+H 不足时）：cross_recon 解耦锚定（v3 修正）
**CEO 声部关键修正**：step2 的 lip_space/pose_space 余弦损失梯度流向 enc/fc/direction — **在默认冻结配置下对 dec 梯度恒为零（死代码）**。因此 A 重述为：
- **only_dec 模式下唯一有效的新增监督 = cross_recon**：移植合成帧作为额外编码输入，dec 必须响应对应的 lip alpha。
- **target 语义（ENG 声部 H3，high）**：合成帧 C=A+嘴B **只做编码输入，重建 target=真帧 B**（step2 原语义：接缝不进 target）。若把 target 设为合成帧本身，dec 会学会复现 paste 接缝伪影（±5px 边界、仿射色差）— 推理时无接缝，等于直接注入伪影。且合成对**仅限姿态近邻**（A/B 姿态几乎相同、口型不同）。
- `lip_space`/`pose_space` **仅在 `--unfreeze_direction` 分支启用**（两种模式损失语义不同，代码与文档写明）；数值行为原样搬运（`torch.exp(-cos)` 无 clamp — 与底模可比性优先，不"改进"）。
- **显存（ENG 声部实测）**：batch 4 三路编码实测 10.65GB（16GB 卡余量充足；实测每样本 1.5-1.6x 而非预估 2-2.5x）。**batch 6 触发 WDDM 分配器雪崩（4.4s/步）— 禁用**。
- `get_IA_MB` 抽取到 `data_preprocess/mouth_transplant.py` 叶子模块（无 torch/lmdb 依赖，`crop_size` 参数化 224→256；原 dataset 原地 import 保持兼容）。
- train_fine_tune.py 残留 SUPPRESS 的 `lip_space_weight/pose_space_weight` 参数：删除或与 A 语义严格对齐 — 禁止"顺手复活"（CLI 语义陷阱）。

### E（P0，保护网先行）：底模行为回归快照
固定输入（source 帧 + alpha 序列）跑底模 ckpt 推理，锁定输出张量统计量（tests/ 快照）。**独立声部评为全计划最防后悔项，升 P0**。

### G（P1，独立缓解层，可与训练线并行）：推理侧 mouth 对齐贴回（v3 重设计）
**ENG 声部 critical 修正：渲染帧的每帧 landmark 不存在** — 现有 landmark 全部来自原图预处理管线；"每帧检测开销可忽略"的前提错误（渲染线程每加 1ms CPU 延迟都直接转化为音频 HOLD）。重设计：
- **首选方案：无检测配准** — 嘴区 ECC/相位相关（rendered_256 vs 原图 crop 嘴区），无 landmark 依赖，CPU 开销 <1ms 量级；
- 备选：N 帧一次检测 + EMA（复用 compositor 既有 `_smoothed_facemark` EMA 纪律）。
- **插入点**：CUDA graph 路径返回静态输出张量 — 必须在 GPU→CPU 下载**之后**、blend **之前**做 warp（现有 precomputed mask 原样有效）。
- **线程纪律**：transform 状态并入 compositor 既有 `_state_lock`（generate_mask 在 preload 线程、blend 回退在渲染线程）。
- **变换钳位**：|scale-1|>0.15 或平移>10px 直接拒绝并回退现行为 — 防止配准失败把嘴贴飞。
- 工作量重估：human 1-2 天 / CC ~2h。**直接消灭症状 3，不依赖任何训练成功**；EMBEDDING 契约面（M1 签名/M23/M4）不触碰，已核实。

### F（P2，暂缓，同前）
- Audio2Lip 个体化（`train/trainer_audio2mouth.py` 全套已在仓库；带"向底模输出正则"则对姿势/背景零风险；边界：不提升嘴部渲染保真）。
- ArcFace ID 损失 / dec 分层冻结 / LoRA 低秩适配（一句评估：可行但引入新依赖，等 A-E 结果）。

## 4. 验收标准（基线校准后修订）

0. **先测当前失败 ckpt 的四指标基线**，阈值按"底模值 × 安全系数"修订（下述为初始草案）；
1. held-out 帧集上姿势保持 MAE ≤ 底模 × 1.5；
2. 同 alpha 下 lip 开合幅度比 0.8-1.25×；
3. 背景 SSIM 相对底模下降 ≤ 0.02（诊断线）；
4. E2E 贴回嘴部 landmark 偏移 ≤ 2px（G 落地后应与姿势漂移解耦）；
5. 主观：发丝/嘴唇纹理不劣于底模。

**时间盒（独立声部补）**：C+H+B 落地后一个训练实验周期（~2 周内）四指标不达标 → 启动 A；A 一个周期仍不达标 → G 兜底 + 复盘训练路线。

## 5. NOT in scope（v2 修订）

- ~~推理侧 mouth 对齐~~ → **移回 P1（方案 G）**（独立声部推翻原判定）；
- Audio2Lip 重训 → 降为 F/P2（工具链已在仓库，边界已注明）；
- 实时管线的其他改动（render_loop 调度/贴回以外的重构）；
- 底模（step1/step2）重训；
- AMP 半精度（Windows 稳定性风险 > 收益）；
- 数据增广（与微调分布一致前提冲突）；
- step2/微调 trainer 架构统一（长期项）。

## 6. What already exists（复用，不重写）

- `get_IA_MB` 嘴部移植（datasets/dataset_HDTF_lip_nonlip.py:208，注意 224/256 适配）；
- step2 的 lip_space/pose_space 实现（trainer_Mouth_Pose_decouple.py:174-177，注意梯度语义）；
- landmark 工具：`ckpts/shape_predictor_68_face_landmarks.dat` + `s3fd` + compositor facemark 管线；
- Audio2Lip 训练全套（train/trainer_audio2mouth.py，F 项用）；
- 契约快照惯例（tests/）→ 底模回归快照；
- `obtain_seq_index` 越界 clamp 模式。

## 7. 实施顺序（v3：数据驱动的决策门 + 显存标定前置）

1. **T1** C 参数修正 + **T2** E 底模快照（保护网，快照钉死 runtime312 解释器 + 记录 torch/cuDNN 版本，环境不符 skip+告警）；
2. **T3** D 评估协议（held-out 排除、自校准+元测试、分组插值）→ **跑当前失败 ckpt 基线**；
3. **显存标定（ENG 声部实测）**：真实数据 50 步记录 `torch.cuda.max_memory_allocated` — **batch8 含 VGG 实测 15.05GB 贴 WDDM 悬崖（2535ms/步 = 3.2x 溢出阶跃指纹，历史崩溃的机制解释）**；batch4 含 VGG 12.03GB（每样本快 ~2.4x）。"耗时阶跃 = 溢出信号"判读法写进 RUNBOOK；训练冒烟固化 pytest（合成数据 2 步 + 显存上限断言），防未来改动静默重回悬崖；
4. **T4** B 采样（近邻+自对+远对≥30%+held-out 排除）+ **T5** H 嘴区加权（背景×1.0、GAN 不加权）→ 训练实验 #1（时长按实测 b4 ~440ms/步 → 30k iter ≈ 4-5h/次，时间盒按此操作化）；
5. 决策门：四指标达标 → 交付；不达标 → **T6** A（cross_recon 锚定，target=真帧）→ 实验周期 #2；
6. **T7** G 推理侧对齐（v3 无检测配准重设计，与 3-5 并行，独立验收症状 3）；
7. 仍不达标 → F 项评估 + 复盘。

**stddev 脚注（ENG 声部发现 12）**：两版 Discriminator 的 minibatch stddev 在 batch=4 时实测完全等价 — R3 的"口径差异"从根因清单降级为脚注，R3 的真实问题就剩单数据 D 过拟合（`--gan_weight` 开关覆盖）。

## CEO REVIEW（plan-ceo-review 全方法论，SELECTIVE EXPANSION，auto-decide）

## CEO Step 0A — 前提挑战

| 前提 | 评估 | 结论 |
|---|---|---|
| P1 微调是提升单角色质量的正确路径 | 替代：换 source 零样本（质量不足）、GFPGAN 后处理（实时开销+不解决姿势）、重训单人架构（数周，推翻管线）；**独立声部补充"不微调"锚点** — 已纳入 D 对比组 | ✅ 接受 |
| P2 推理时 pose drive ≡ source 自身（render_loop.py:727） | 代码证实；未来段切换若改传不同帧，方案 B 采样需跟随 | ✅ 接受，标注依赖 |
| P3 微调数据与推理同 crop 分布 | 假设合理；预处理落地时验证 | ✅ 接受，带验证步骤 |
| P4 姿势漂移主因在训练侧 | 差异变量法成立；**但缓解层不必在训练侧** → G 移回 P1 | ✅ 接受 + 缓解层解耦 |
| P5 单人数据可承载 step2 式解耦监督 | **独立声部证伪了一半**：余弦损失在冻结配置下是死代码；A 重述为 cross_recon 锚定 | ⚠️ 已修正 |

## CEO Step 0C — Dream State

```
CURRENT                        THIS PLAN                        12-MONTH IDEAL
微调破坏解耦：姿势漂移    --->  便宜杠杆止血+可信评估    --->  一键角色适配流水线：
背景变形、贴回错位；            +分布对齐+条件触发锚定；         采集→预处理→微调→自动评估
微调路径不可用；                推理侧对齐兜底；                 →评分门限→热切换上线；
质量受限于底模 zero-shot        数据驱动的决策门                 多角色库，<1h/角色
```

## CEO Step 0C-bis — 实施备选方案

| 方案 | 摘要 | Effort | Risk | 取舍 |
|---|---|---|---|---|
| **Alt-1（选定）** 分层数据驱动（C+H+B+D+E 先行，A 条件触发，G 并行） | 便宜杠杆→可信基线→条件触发的锚定工程 | human 3-5天 / CC 6-8h + 实验周期 | Med | 消融隔离根因，避免烧单卡周期 |
| Alt-2 纯调参（只 C+B） | 最快 | S | 大概率不达标（用户已试近似配置） | 保留为对照 |
| Alt-3 换架构（单人 avatar 类） | 上限高 | XL | High（推翻管线） | 拒绝 |
| Alt-4 不微调 + 后处理 | 零训练 | S | 不解决姿势 | 保留为 D 的对比锚点 |

## CEO Step 0D/0E — SELECTIVE EXPANSION 裁决与时间轴

扩展裁决：TensorBoard 分项 → 纳入 D；一键角色适配管线 → DEFER TODOS P2；多角色回归套件 → DEFER P3；AMP → SKIP；数据增广 → SKIP。
时间轴：HOUR 1 C+E；HOUR 2-3 D+基线；HOUR 4-5 B+H+实验#1；HOUR 6+ 决策门 → A 或交付；G 全程并行。

## CEO Sections 1-10 发现（要点）

- **S1** FINDING-CEO1（P1）：landmark 检测失败帧策略未定义 → 预处理失败清单 + dataset 回退 + >10% 中止（CEO3）。
- **S4** FINDING-CEO5（P1）：近邻对上嘴部移植退化（口型几乎相同 → 合成帧≈原图 → cross 无信号）→ **移植对与重建对采样路径分离**（移植对要求口型差阈值）。
- **S4** FINDING-CEO4（P2）：单帧回退与 clamp 在 B 重写时必须保留。
- **S5** FINDING-CEO6（P3）：step2 trainer 的 200 行复制反模式 → 微调用单一 forward 路径。
- **S6** FINDING-CEO7（P1）：eval 自校准基线必须（已纳入 D）。
- **S7** FINDING-CEO8（P2）：三路编码显存 → batch 4 冒烟。
- **S8** FINDING-CEO9（P2）：新损失 tensorboard 分项 + eval JSON 归档。
- **S10** FINDING-CEO10（P3）：landmark 缓存 schema version。
- **S3 安全 / S9 部署**：无发现（本地管线 / ckpt 产物解耦易回滚）。

## CEO Failure Modes Registry

| CODEPATH | FAILURE MODE | RESCUED? | TEST? | USER SEES? | LOGGED? |
|---|---|---|---|---|---|
| landmark 预处理 | 检测失败 | Y（跳过+清单+10%门） | 单测 | 失败率日志 | Y |
| 移植对采样 | 近邻移植退化 | Y（分离采样路径） | 单测 | 否（已拦截） | Y |
| trainer 三路编码 | CUDA OOM | Y（batch 4 冒烟前置） | 冒烟 | OOM 报错 | Y |
| eval 协议 | 指标无参照/训练帧过拟合 | Y（自校准+held-out） | 校准测试 | 对比报告 | Y |
| 底模推理路径 | 代码改动破坏底模 | Y（快照测试） | 快照 | 部署前拦截 | Y |
| 微调实验失败 | 全盘回退颗粒无收 | Y（权重插值扫描） | eval | 部分收益保留 | Y |

CRITICAL GAP：0。

## CEO Implementation Tasks（v2 重排）

- [ ] **T1 (P1, CC ~5min)** — train_fine_tune.py — C 参数修正（lr 0.0001、direction_lr_ratio 1.0、ortho 0、gan_weight 开关、帮助文本对齐）
- [ ] **T2 (P1, CC ~20min)** — tests/test_base_model_snapshot.py — E 底模行为回归快照
- [ ] **T3 (P1, CC ~1.5h)** — tools/eval_finetune.py — D 四指标 + held-out 集划分 + 自校准 + 插值扫描 + JSON 归档；跑当前失败 ckpt 基线
- [ ] **T4 (P1, CC ~30min)** — fine_tune/dataset.py — B 双采样路径（近邻/自对/远对 + 移植对分离 + 回退/clamp）
- [ ] **T5 (P1, CC ~30min)** — trainer 损失 — H 嘴区加权（vgg/l1/gan 嘴区 ×4 / 脸部 ×1 / 背景 ×0.5）+ tensorboard 分项
- [ ] **T6 (P1, CC ~1.5h，条件触发)** — fine_tune/trainer_fine_tune.py — A cross_recon 锚定（移植合成帧 + 仅 unfreeze 分支启用余弦损失 + 单一 forward + batch 4 冒烟）
- [ ] **T7 (P1, CC ~1h)** — realtime_serve/compositor.py — G 推理侧 mouth 相似变换对齐贴回（独立验收症状 3；注意 EMBEDDING 契约）
- [ ] **T8 (P2, CC ~20min)** — RUNBOOK 微调章节（冻结语义/损失语义/评估流程/时间盒）+ landmark 预处理脚本（含失败清单/schema version/10% 门）

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|-----------|-----------|----------|----------|
| 1 | CEO | 模式 = SELECTIVE EXPANSION | Mechanical | autoplan 覆盖规则 | 修缺陷型任务 | — |
| 2 | CEO | Alt-1 分层数据驱动为主菜 | Mechanical | P1+P2 | 独立声部挑战后重排：先便宜杠杆+可信基线 | Alt-3 换架构、Alt-4 后处理为主 |
| 3 | CEO | **A 重述**：余弦损失在冻结配置下是死代码，A 降为 cross_recon 锚定且条件触发 | Mechanical（技术事实验证） | P5 显式 | 梯度路径分析确认：directions 对 dec 是常量输入 | 原方案 A 照搬 |
| 4 | CEO | **G 推理侧对齐移回 P1** | Taste → gate 呈现 | P2 boil lakes | 1 天工作量直接消灭症状 3，不依赖训练 | 维持 NOT in scope |
| 5 | CEO | **H 嘴区加权损失纳入 P0** | Taste → gate 呈现 | P2 | 训练目标对齐产品消费面，成本 1/10 | — |
| 6 | CEO | 权重插值回退 + held-out + 时间盒纳入 | Mechanical | P1 | 零成本保险与可信验收 | — |
| 7 | CEO | B 升 P0 主项（近邻+自对） | Mechanical | P1 | 训练/推理 pose 条件错位的最高杠杆修复 | — |
| 8 | CEO | 一键适配管线 DEFER TODOS P2；AMP/数据增广 SKIP | Mechanical | P3/P5 | 见 0D | — |

# ENG REVIEW（plan-eng-review 全方法论，FULL_REVIEW，auto-decide）

## Eng Step 0 — Scope Challenge

1. **子问题 → 现有代码**：全部映射到位（计划第 6 节 + CEO 0B），无重建。
2. **最小变更集**：首批实际落地 = T1/T2/T3/T4/T5 + T7（A 条件触发、T8 预处理随首批）≈ 7-8 文件。
3. **复杂度检查**：触及 ~8 文件（train_fine_tune.py、fine_tune/{trainer,dataset}.py、tools/{eval,preprocess}.py 新、tests/ 新、compositor.py、RUNBOOK）。触发线边缘 — auto-decide（P2 boil lakes）：全部在 blast radius 内且 A 条件触发削减了首批规模，**按原样推进**。新类 ≤2（eval 引擎 + 预处理 CLI）。
4. **Search check**：Search unavailable — in-distribution 知识。计划全部使用既有模式（StyleGAN2 微调、landmark 预处理、快照测试），无新基础设施消费 innovation token。
5. **TODOS 交叉**：TODOS.md 现有项（音频感知切换/gaze 懒初始化/多角色切换等）与本计划零阻塞、零捆绑。新增 3 项 TODO（见 TODOS 更新节）。
6. **完整性**：D 协议完整（4 指标+held-out+自校准+插值）；测试全覆盖矩阵见 S3。
7. **分布检查**：无新 artifact 类型（本地训练/评估脚本）。N/A。

## Eng Section 1 — Architecture

```
                 训练线（fine_tune/ + tools/）
 帧目录(count/) ─▶ tools/preprocess_finetune_data.py [T8]
                    │  产出缓存: landmarks/{frame}.npy (schema v1)
                    │           mouth_syn/ 移植合成帧缓存 [离线!见E1]
                    │           val_split.json (held-out, D)
                    │           preprocess_failures.txt (清单, >10%中止)
                    ▼
 train_fine_tune.py [T1 参数] ─▶ fine_tune/dataset.py [T4: 近邻/自对/远对/移植对索引]
                                     │ 只做采样，不做在线仿射
                                     ▼
                          fine_tune/trainer_fine_tune.py [T5: vgg/l1 嘴区加权]
                                     │            [T6: cross_recon@条件, 余弦损失@unfreeze分支]
                                     ▼
                          ckpt_models/{exp}/checkpoint/*.pt
                                     │
                                     ▼
 tests/test_base_model_snapshot.py [T2]   tools/eval_finetune.py [T3: 四指标+held-out
        (底模统计量锁定)                    +自校准+插值扫描(仅dec)+JSON归档]
                 ─────────────────────────────────────────────
                 推理线（realtime_serve/，G 独立 lane）
 render_loop.py ─▶ compositor.py [T7: blend 内相似变换对齐; composite()签名不变]
                    （EMBEDDING 契约面=M1签名/M23热切/M4错误包络 — 不触碰，已核实）
```

- **发现 E1（P2, confidence 8/10）**：移植仿射必须**离线缓存**而非在线计算 — get_IA_MB 每样本 ~10ms CPU 仿射在 Windows spawn + num_workers=2 下成瓶颈，且移植对需口型差筛选（CEO5）在线随机采样无法保证。→ 架构决策：预处理产合成帧缓存，dataset 只做索引。已并入 T4/T8 职责。
- **发现 E3（P2, confidence 8/10）**：held-out 用时间切分（末 10%）存在相邻帧泄露 — 训练/评估帧内容近似，泛化被高估。**决策：保留时间切分**（本用例推理分布本就是同角色近似姿势，泄露方向与真实用途一致），在 eval 输出中标注该局限。跨视频划分对单人数据不可行。
- **发现 E4（P3, confidence 7/10）**：插值扫描只对 dec 有意义；unfreeze_direction 模式下 direction.weight 插值经 QR 非线性，中间点行为不可预测。→ eval 限定：插值仅 dec，unfreeze 模式跳过 direction 插值。
- 回滚 posture：训练线 git revert；缓存目录独立可删；G 有"对齐关闭=旧行为"开关（见 S3 回归铁律）。
- 生产失败场景：预处理中止（失败率门）→ 用户看到明确清单文件；训练 OOM → 冒烟前置；G 对齐失败（检测不到脸）→ 回退原 mask + 日志。

## Eng Section 2 — Code Quality

- **发现 E2（P2, confidence 9/10）**：**"GAN 嘴区加权"字面不成立** — Discriminator 是全图判别器（networks/discriminator.py:202 起，256 全图输入），无法按区域加权。→ H 修正为：**vgg/l1 按区域加权（multi-scale pyramid mask 对齐），GAN 保持全图**。patch 判别器方案拒绝（新基础设施，innovation token；留 TODO）。
- **发现 E6（P3, confidence 8/10）**：搬运 lip_space/pose_space 时不要"改进"数值（torch.exp(-cos) 无 clamp）— 底模原样收敛过，与底模可比性优先。T6 注记。
- **发现 E7（DRY 决策）**：H 的嘴区 mask 与 compositor._create_mouth_mask 用途不同（训练加权 vs 贴回 alpha 过渡带）— **不强行共享**，训练侧用简化凸包 fill；共享会跨线耦合。记录理由，非违规。
- step2 trainer 200 行复制反模式（CEO6）→ 微调单一 forward 路径。已有。
- 既有 ASCII 图审计：compositor.py 头注释 mask 策略图在 G 改动后需同步更新（diagram maintenance）。

## Eng Section 3 — Test Review（测试框架：pytest，tests/ 既有惯例）

```
CODE PATHS（计划新增/修改）                          测试计划
[+] tools/preprocess_finetune_data.py [T8]
  ├── landmark 成功→缓存 .npy+schema v1 ────────── [GAP→单测] mock 检测器
  ├── 检测失败→跳过+清单 ───────────────────────── [GAP→单测]
  ├── 失败率>10%→中止 ──────────────────────────── [GAP→单测]
[+] fine_tune/dataset.py [T4]
  ├── 时序近邻 ±K / 自对 5-10% / 远对 20% ───────── [GAP→单测] seed 固定验证分布
  ├── 单帧回退（现有分支回归保护）───────────────── [GAP→单测] CRITICAL
  ├── 越界 clamp ───────────────────────────────── [GAP→单测]
  └── 移植对索引（A 启用时）────────────────────── [GAP→单测]
[+] trainer [T5+T6]
  ├── 嘴区加权 vgg/l1（pyramid mask 对齐）───────── [GAP→单测] 权重和校验
  ├── GAN 不加权（E2 修正）─────────────────────── [GAP→单测]
  ├── cross_recon 前向（A）+ 损失有限 ───────────── [GAP→单测] tiny config
  ├── only_dec 梯度只到 dec（autograd 断言）─────── [GAP→单测] 冻结语义回归
  └── 余弦损失仅 unfreeze 分支 ──────────────────── [GAP→单测]
[+] tests/test_base_model_snapshot.py [T2]
  └── 底模输出统计量锁定 ────────────────────────── [GAP→快照] skip-if-no-cuda
[+] tools/eval_finetune.py [T3]
  ├── 自校准（底模vs底模=0）────────────────────── [GAP→校准测试]
  ├── 指标计算（合成数据已知答案）───────────────── [GAP→单测]
  └── 插值扫描仅 dec ───────────────────────────── [GAP→单测]
[+] realtime_serve/compositor.py [T7]
  ├── 相似变换对齐恢复合成偏移 ──────────────────── [GAP→单测] numpy 确定性
  ├── 对齐失败回退旧 mask 路径 ──────────────────── [GAP→单测] CRITICAL（回归铁律）
  └── 对齐关闭=与旧版行为逐像素一致 ─────────────── [GAP→回归] 现有 test_t12/t13 必须全绿

覆盖: 0/19 有测试（全新路径）| GAPS: 19（全部纳入实施）| 2am Friday 测试 = E 快照 + G 关闭路径回归
```

REGRESSION 铁律：T7 改 compositor 现有行为 → "对齐关闭路径与旧版一致"回归测试为 CRITICAL 必做；tests/test_t12_compositor.py、test_t13_compositor_integration.py 落地前后各跑一次。

## Eng Section 4 — Performance

- **发现 E8（P2, confidence 7/10，medium—冒烟验证）**：batch 4 三路编码 + 2-3 路 dec 前向在 4080S 16GB 的余量未知 — 冒烟前置；不足则梯度累积 ×2。feats_t 自重建分支无梯度回传需求（enc 冻结）无需额外 detach 处理。
- 预处理 O(N) 一次性分钟级；eval 5λ×双模型推理分钟级。无 N+1/DB 类问题。
- WDDM：既有 wddm_sync_interval 机制保留，训练循环结构不变。

## Eng — NOT in scope（补充确认）

同计划第 5 节 + patch 判别器（E2 拒绝项 → TODOS P3）。

## Eng — What already exists

无重复建设（计划第 6 节 + CEO 0B 已映射；E1 决策使预处理复用 landmark 工具链而非新建检测）。

## Eng Failure Modes Registry

| CODEPATH | FAILURE | RESCUED? | TEST? | USER SEES? | LOGGED? |
|---|---|---|---|---|---|
| 预处理 | landmark 失败率高 | Y（10% 门中止） | 单测 | 清单文件+明确中止 | Y |
| dataset | 单帧/越界 | Y（回退+clamp） | 单测 | 否 | N/A（无日志面） |
| trainer | OOM | Y（冒烟+梯度累积 fallback） | 冒烟 | 报错 | Y |
| G 对齐 | 渲染帧无脸可对齐 | Y（回退旧 mask） | 单测 | 无感回退 | Y |
| eval | 指标误导 | Y（自校准+held-out 标注） | 校准测试 | 报告含局限注记 | Y |

CRITICAL GAP：0。

## Eng — Worktree 并行化策略

| Step | 模块 | 依赖 |
|---|---|---|
| T1 参数 | train_fine_tune.py | — |
| T2 快照 | tests/ | — |
| T3 eval | tools/, tests/ | T2（快照复用加载器） |
| T8 预处理 | tools/ | — |
| T4 采样 | fine_tune/ | T8 缓存 schema |
| T5 加权 | fine_tune/ | — |
| T6 cross_recon | fine_tune/ | T8+T4 |
| T7 对齐 | realtime_serve/ | — |

**Lane A（训练线）**：T8 → T4 → T6（+T1/T5 顺路）— 顺序，共享 fine_tune/。
**Lane B（评估线）**：T2 → T3 → 失败 ckpt 基线跑批 — 独立。
**Lane C（推理线）**：T7 — 完全独立。
执行：A+B+C 三 lane 并行 worktree；A 与 B 只在"基线跑批需要 B+H 训练产物"处汇合。

## Eng — TODOS 更新（收集 CEO+Eng 全部 defer，auto-decide 写入 TODOS.md）

1. 一键角色适配管线（CEO DEFER，P2）
2. 多角色微调回归套件（CEO DEFER，P3）
3. patch 判别器嘴区 GAN（Eng E2 拒绝项留档，P3）

## Eng Implementation Tasks

（合并进下方统一任务表 T1-T8；Eng 新增细化：T7 的回归铁律、T8 的 10% 门与 schema version、T4 的移植对索引。）

## ENG Step 0.5 — 双声部（Codex 不可用 → [subagent-only] 降级）

### CLAUDE SUBAGENT (eng — independent review) 摘要

独立声部用 runtime312 实测了 ckpt 加载（gen 195 张量 / dis 50 张量 / weight 512×26）并跑了**合成数据真实前向+反向显存冒烟**。整体判断：**方向正确，需修正后执行**。4 项开工前必修：

1. **发现 G-critical**：渲染帧每帧 landmark 不存在（现有 landmark 全来自原图预处理）— G 的"每帧检测开销可忽略"前提错误 → v3 已重设计为无检测配准（ECC/相位相关）+ 插入点/线程纪律/变换钳位，工作量重估 1-2 天。
2. **E1 held-out 泄漏（high）**：B 的 ±64 邻域会采到 held-out 帧当 target — 全部采样路径显式排除 + 单测断言 → v3 已并入 B。
3. **H1+H2（high）**：GAN 结构上无法按区加权（全图单 logit）写死不加权；背景 ×0.5 反向放松 direction 膨胀约束 → 改 ×1.0 → v3 已修正。
4. **H3（high）**：cross_recon 的 target 必须是真帧 B（合成帧只做编码输入），且仅姿态近邻对 — 否则 paste 接缝伪影进 dec → v3 已写进 A。
5. **显存实测（high，计划外发现）**：现状 batch 8 已 13.66GB（未含 VGG）贴 WDDM 悬崖 — 历史崩溃的合理机制；batch 4 实测 10.65GB 确认可行；batch 6 分配器雪崩（4.4s/步）禁用 → 显存标定步骤进实施顺序。
6. **A1（medium）**：get_IA_MB 抽 `data_preprocess/mouth_transplant.py` 叶子模块（该目录已存在），crop_size 参数化。
7. **A2（medium）**：facemark npy 与帧序的耦合是巧合维持 — mask 缓存 fail-fast 校验（len+stem 连续），NaN 行回退全 1 权重。
8. **T1 eval 元测试（high）**：敏感性/确定性/NaN 三类元测试；compositor EMA 有状态必须重置。
9. **T2 快照钉环境**：runtime312 唯一权威解释器 + torch/cuDNN 版本记录。
10. **E3（medium）**：近邻对收缩 lip alpha 方差 → 远对 ≥30% + Audio2Lip alpha 直混样本可选。
11. **H4（medium）**：direction 插值 QR 非线性 → 分组插值。
12. **发现 12（low）**：stddev 口径 batch=4 实测完全等价 — R3 降级脚注。
13. **发现 13（low）**：SUPPRESS 的 lip_space_weight/pose_space_weight 禁止复活。
14. **S1（medium）**：weights_only=False 的 pickle 面 — RUNBOOK 标注"只加载自产 ckpt"；eval/快照走纯 state_dict + weights_only=True。

### ENG DUAL VOICES — CONSENSUS TABLE [subagent-only]

```
═══════════════════════════════════════════════════════════════
  Dimension                            Claude  Sub   Consensus
  ──────────────────────────────────── ─────── ───── ─────────
  1. Architecture sound?                ✅      ✅+A1/A2 CONFIRMED（修正后）
  2. Test coverage sufficient?          ⚠️      ❌→✅   DISAGREE→修正采纳（元测试/钉环境）
  3. Performance risks addressed?       ⚠️标注  ✅实测  CONFIRMED（实测强化：b8悬崖+b4可行）
  4. Security threats covered?          ✅      ⚠️S1   PARTIAL→RUNBOOK 注记
  5. Error paths handled?               ✅      ❌→✅   DISAGREE→修正采纳（NaN/eval失败路径）
  6. Deployment risk manageable?        ✅      ❌→✅   DISAGREE→修正采纳（G 重设计）
═══════════════════════════════════════════════════════════════
  Codex voice: N/A（auth 缺失，[subagent-only] 降级）
  6 维中 2 CONFIRMED、4 DISAGREE — 全部经 v3 修订吸收；G 项 critical 前提
  错误的重新设计为最重要产出。显存实测把"batch 8 崩溃史"从玄学变成机制。
```

**PHASE 3 COMPLETE.** Codex: N/A（不可用）。Claude subagent: 14 项发现（1 critical + 5 high）。
Consensus: 2/6 confirmed 原样，4 处分歧全部经 v3 修订吸收。
Passing to Phase 4 (Final Gate)。

## COMPLETION SUMMARY（CEO + ENG）

```
+====================================================================+
|        MEGA PLAN REVIEW — COMPLETION SUMMARY (via /autoplan)       |
+====================================================================+
| Mode                | SELECTIVE EXPANSION (auto)                    |
| System Audit        | ckpt/网络/数据集/推理链全量走查 + 实测加载验证  |
| Step 0              | Alt-1 分层数据驱动；scope 保持 + G/H 移入      |
| Section 1  (Arch)   | 4 issues (E1-E4) — 全部折叠为架构决策          |
| Section 2  (Errors) | 6 failure paths mapped, 0 GAPS                 |
| Section 3  (Security)| 2 findings (S1 pickle 面/S2 路径) — low 风险   |
| Section 4  (Data/UX)| 5 edge cases mapped, 0 unhandled (v3 后)       |
| Section 5  (Quality)| 3 findings (反模式/数值搬运/DRY 决策)          |
| Section 6  (Tests)  | 图 produced, 19 gaps → 全部纳入任务            |
| Section 7  (Perf)   | 2 findings (显存实测 b8悬崖/b4可行)            |
| Section 8  (Observ) | 2 findings (tensorboard 分项/JSON 归档)        |
| Section 9  (Deploy) | 0 risks (ckpt 产物解耦，回滚简单)               |
| Section 10 (Future) | Reversibility 4/5, debt items: 3 (TODOS)       |
| Section 11 (Design) | SKIPPED (no UI scope)                          |
+--------------------------------------------------------------------+
| NOT in scope         | written (8 items)                             |
| What already exists  | written                                       |
| Dream state delta    | written                                       |
| Error/rescue registry| 6 methods, 0 CRITICAL GAPS                    |
| Failure modes        | 11 total, 0 CRITICAL GAPS                     |
| TODOS.md updates     | 3 items written                               |
| Scope proposals      | 5 proposed, 2 accepted (G/H), 2 deferred, 1 skipped |
| CEO plan             | n/a (SELECTIVE, 修复型任务)                    |
| Outside voice        | ran (claude subagent ×2) / codex unavailable   |
| Lake Score           | 12/12 recommendations chose complete option    |
| Diagrams produced    | 3 (dream state / 训练+推理架构 / 测试覆盖)      |
| Stale diagrams found | 1 (compositor.py mask 策略注释 — T7 同步更新)   |
| Unresolved decisions | 2 taste decisions → Final Gate (G/H scope)     |
+====================================================================+
```

## Cross-Phase Themes（跨阶段主题）

**Theme 1: "把 step2 结构照搬"的两面性** — CEO 声部发现余弦损失死代码（critical），ENG 声部发现 target 接缝伪影（high）。两个声部独立指向同一主题：**搬运底模监督结构时，梯度语义与 target 语义必须逐项重新推导，不能照搬**。高置信信号（两阶段独立确认）。
**Theme 2: 实测代替估算** — CEO 声部实测 ckpt 加载，ENG 声部实测显存（b8 悬崖/b4 可行/b6 雪崩）。计划中所有资源类断言已全部替换为实测数字。
**Theme 3: 静默失败是最大风险面** — held-out 泄漏（E1）、NaN landmark（E2）、mask 错位（A2）、移植退化（CEO5）全部是"不报错只变差"类。对策统一：fail-fast 校验 + 单测断言 + 日志计数。

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/autoplan` Phase 1 | 战略/范围/根因 | 1 | issues_open→gate | 8 发现折叠 + 独立声部 7 项（1 critical 采纳：余弦损失死代码） |
| Codex Review | `/autoplan` Phase 0.5 | 独立第二意见 | 0 | N/A | Codex 不可用（auth 缺失），降级 Claude subagent 双阶段 |
| Eng Review | `/autoplan` Phase 3 | 架构/测试/性能（必需门） | 1 | issues_open→gate | 14 项独立声部发现（1 critical：G 项 landmark 前提错误）+ 显存实测 |
| Design Review | — | 无 UI scope | 0 | SKIPPED | 训练管线，无 UI |
| DX Review | — | 非 developer-facing 产品 | 0 | SKIPPED | 内部 ML 管线 |

- **CROSS-MODEL:** 两声部（CEO+Eng）独立确认 R1 机制与消费面框架；跨阶段主题 3 个（照搬结构需逐项重推语义 / 实测代替估算 / 静默失败风险面）。
- **VERDICT:** CEO + ENG 双阶段完成，全部 critical/high 修正已折叠进 v3 — 待 Final Gate 批准。

**UNRESOLVED DECISIONS:**
（2026-09-04 Final Gate：用户批准 A — 按推荐执行，含 G/H 两项 taste 推荐 → 全部解决）

NO UNRESOLVED DECISIONS

## Final Gate 裁定（2026-09-04）

- **批准**：v3 计划按推荐执行（T1→T8 顺序，G/H 两项 taste 推荐一并批准）。
- Eng 声部最终版补充修正（已吸收）：batch_size 默认 8→4（T1）；单帧视频回退是真 bug（dataset.py:108-113 `random.sample(range(1),2)` 崩溃 — T4 修复项）；移植对嘴区 bbox 触界 → 弃用进失败清单（比 clamp 安全）；H 的 mask 语义直接复用 `compositor._create_mouth_mask` 双边界（训练权重面 ≡ 产品消费面 + 少一份实现）；`trainer.save` 的 args 改存 dict（为 weights_only=True 铺路，RUNBOOK 标注"只加载自产 ckpt"）。

## CEO Step 0.5 — 双声部（Codex 不可用：auth 缺失 → [subagent-only] 降级）

### CLAUDE SUBAGENT (CEO — strategic independence) 摘要

独立声部用仓库 runtime312 实际加载 ckpt 做了严格验证（gen strict=True 加载成功、dis 50 张量零 missing/unexpected），逐条确认事实底座后给出 **整体判断：需修正**：

1. **发现 1（critical）**：lip_space/pose_space 余弦损失梯度流向 enc/fc/direction — 方案 A 默认冻结这些组件时损失是死代码。→ 已采纳：A 重述为 cross_recon 锚定，余弦损失仅 unfreeze 分支启用。
2. **10x 重新框架**：compositor 只贴回嘴区 — 背景变形不是产品输出问题（保留为诊断指标）；三个低成本杠杆（嘴区加权、推理侧对齐、Audio2Lip 个体化）被遗漏。→ 已采纳：H 纳入 P0、G 移回 P1、F 项降 P2 注明边界。
3. **发现 2（high）**：R1/R2/R3 无消融隔离，A 定 P0 早于任何实验 → 烧单卡周期。→ 已采纳：重排 C→E→D 基线→B+H→决策门→A 条件触发。
4. **发现 3（high）**：机制描述修正 — dec 无 feats_t 可抄；真实机制是"训练时 pose alpha 恒来自 target 帧 → dec 重写 pose 响应绑定；推理时 pose alpha 来自 source 自身的恒等条件从未被显式训练"。→ 已采纳：2.3 重写；B 升 P0 主项 + 混入同帧自对。
5. **发现 4（medium）**："pose direction 恒定"表述不准（实为小幅头动序列）；get_IA_MB 的 224×224 输出与 256 空间越界风险；R3 无动作 → 已采纳：表述修正、T6 注记、`--gan_weight` 开关。
6. **风险盲区**：held-out 缺失（dataset 无 val split）、阈值无基线校准、时间盒缺失、无回退保险 → 已采纳：D 升 P0 先跑失败 ckpt 基线 + held-out + 插值扫描；时间盒写入第 4 节。
7. **肯定项**：E（底模快照）评为全计划最防后悔项 → 升 P0；T1 参数修正属实 10 分钟可完成。

### CEO DUAL VOICES — CONSENSUS TABLE [subagent-only]

```
═══════════════════════════════════════════════════════════════
  Dimension                            Claude  Sub   Consensus
  ──────────────────────────────────── ─────── ───── ─────────
  1. Premises valid?                    ✅修正后 ✅     CONFIRMED（发现1采纳后）
  2. Right problem to solve?            ✅      ✅+重构 CONFIRMED（消费面框架采纳）
  3. Scope calibration correct?         ⚠️      ❌→✅   DISAGREE→修正采纳（G/H 移回）
  4. Alternatives sufficiently explored? ⚠️     ❌→✅   PARTIAL→补齐（插值/锚点/开关）
  5. Risks covered?                     ⚠️      ❌→✅   DISAGREE→修正采纳（held-out/时间盒）
  6. 6-month trajectory sound?          ✅      ❌→✅   DISAGREE→修正采纳（死损失/排序）
═══════════════════════════════════════════════════════════════
  Codex voice: N/A（auth 缺失，[subagent-only] 降级）
  6 维中 2 CONFIRMED、4 DISAGREE — 全部经 v2 修订吸收；关键分歧升格为
  Taste Decisions（#4 G 移回 scope、#5 H 纳入 P0）在 Final Gate 呈现。
```

**PHASE 1 COMPLETE.** Codex: N/A（不可用）。Claude subagent: 7 项发现（1 critical）。
Consensus: 2/6 confirmed 原样，4 处分歧全部经 v2 修订吸收 → gate 呈现 2 项 taste。
Passing to Phase 3（Phase 2 Design 跳过 — 无 UI scope；Phase 2.5 DX 跳过 — 非 developer-facing 产品）。
