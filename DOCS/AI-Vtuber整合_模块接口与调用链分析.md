# AI-Vtuber 整合 · 模块接口与调用链分析

> **基线**：tag `integrate-prep-20260921`（= commit `f220a6e`，main）。本文档所有 file:line 与行为描述均以该基线为准；EDTalk 后续变更请在「§14 变更登记」补记，防止文档静默腐烂。
> **工作区漂移防护（Eng A2）**：基线 SHA 锚定的是已提交状态；整合实施时若工作区存在未提交变更（当前 fine_tune/、train/、data_preprocess/、realtime_serve 均有），**实施前先执行 T1（收尾提交 + 打 tag `integrate-prep-20260921`）**，以 tag 为实施起点。工作区漂移不构成契约变更——接口以 tag 时刻为准，变更走 §14 登记。
> **契约单一来源声明（Eng A3）**：realtime_serve 方法面/HTTP 契约以 `realtime_serve/EMBEDDING.md`（v0.3，冻结）为单一事实来源；本文 §3 是整合视角的补充与差异登记（§3.3）。**两者冲突时：§3.3 已登记的差异以本文为准，其余一律以 EMBEDDING.md 为准**——禁止在第三处文档新增接口事实。
> **fastrag 分析边界（Eng A4）**：fastrag 为已编译 Rust 二进制，本文只覆盖**进程边界**（HTTP/文件/环境变量/退出码）可分析面；内部行为（推理细节、异常返回格式）按黑盒对待，行为级差异发现时登记进 §14。
> **用途**：EDTalk 四个模块整合进 AI-Vtuber 的实施依据。深度约定：写"接缝"（参数/协议/依赖方向/产物路径），不逐行复述内部实现。机制细节见 DOCS/ 现有文档（§15 索引）。

---

## 目录

1. [备选方案与取舍记录](#1-备选方案与取舍记录)
2. [系统总览与速查表](#2-系统总览与速查表)
3. [realtime_serve 实时推理服务](#3-realtime_serve-实时推理服务)
4. [fastrag 知识库问答服务](#4-fastrag-知识库问答服务)
5. [数据预处理管线（11 步）](#5-数据预处理管线11-步)
6. [链路A：底模微调 train_fine_tune.py](#6-链路a底模微调-train_fine_tunepy)
7. [链路B：Audio2Lip 微调 train/train_audio2mouth.py](#7-链路baudio2lip-微调-traintrain_audio2mouthpy)
8. [一键训练编排蓝图](#8-一键训练编排蓝图)
9. [配方与 checkpoint 流转（一等公民）](#9-配方与-checkpoint-流转一等公民)
10. [静默失败登记（高危）](#10-静默失败登记高危)
11. [与 AI-Vtuber 的对接面汇总](#11-与-ai-vtuber-的对接面汇总)
12. [整合验收清单](#12-整合验收清单)
13. [附录A：视频通道延迟预算](#13-附录a视频通道延迟预算)
14. [变更登记](#14-变更登记)
15. [附录B：引用文档索引](#15-附录b引用文档索引)

---

## 1. 备选方案与取舍记录

| # | 决策点 | 选择 | 被否方案 | 一句话理由 |
|---|---|---|---|---|
| 1 | 整合前是否先写接口文档 | **写文档先行**（用户指令） | 直接按对接纪要 A1-A4 开工 | 四模块接口散落 4+ 处文档与记忆，实施时需反复考古；本文档为单一视图 |
| 2 | 文档深度 | **接缝契约优先**，调用链按接口层组织 | 全量内部实现链路复写 | 全量文档 6 个月后必然腐烂；接缝（参数/协议/路径）才是一直需要的 |
| 3 | 视频通道 | **虚拟摄像头（UnityCapture DirectShow）**——已裁决先例（2026-09-03 对接纪要） | 共享内存/帧 API | 崩溃域隔离、OBS 生态兼容；延迟预算见附录A，不重开 |
| 4 | 音频通道 | **HTTP push_full（整句推送）** | 流式 /audio/push 分块 | 实现中流式路径无消费者（§10-1），整句推理唇形最稳；TTS 16kHz 已确认无需转码 |
| 5 | 服务拓扑 | **各服务独立进程 + 独立端口，中枢 HTTP 调用**（统一环境计划 §6 已裁决） | 进程内直调/合并 | 崩溃域隔离；同机回环零网络开销 |
| 6 | 一键训练形态 | **编排蓝图先行（本文档 §8），脚本属实施阶段** | 本轮直接写编排脚本 | 编排依赖训练线验收达标（TODOS P2 前置条件）；先冻结流程再固化代码 |
| 7 | fastrag 子模块化 | **git init + 模型排除**（4.7GB 走独立分发） | 全量入库 / LFS | 模型非代码资产，git 管代码与配置；LFS 对直播机本地收益低 |
| 8 | 可执行契约（对接冒烟脚本） | **延期 TODOS** | 本轮实现 | 依赖整合实施启动；验收清单（§12）是其种子 |

---

## 2. 系统总览与速查表

### 2.1 整合架构图

```
                        ┌─────────────────────────────────────────────┐
                        │              AI-Vtuber（直播中枢，:8086 webui）│
                        │   main.py 编排 / comment_handle 指令责任链     │
                        │   playback_manager（TTS 音频播放点）           │
                        └──────┬───────────────┬───────────────┬──────┘
                               │ HTTP          │ HTTP          │ HTTP
              ┌────────────────▼───┐   ┌───────▼────────┐  ┌───▼──────────────┐
              │ GPT-SoVITS (TTS)   │   │ EDTalk          │  │ fastrag (Rust)   │
              │ :9880              │   │ realtime_serve  │  │ server.exe :11420│
              │ 16kHz wav 输出     │   │ :8000 FastAPI   │  │ RAG 问答          │
              └─────────┬──────────┘   └───────┬─────────┘  └──────────────────┘
                        │ wav                  │ ①音频 push_full
                        │                      │ ②控制 segment/gaze/repair
                        ▼                      ▼ ③状态 /status
              ┌──────────────────┐   ┌─────────────────────────────┐
              │ 音频双路分发       │   │ 16kHz int16 PCM → Audio2Lip │
              │ 本地播放+推送EDTalk│   │ → Generator → 视频帧         │
              └──────────────────┘   │ → UnityCapture 虚拟摄像头     │
                                     │   → OBS/直播推流端消费        │
                                     │ （音频由服务经 sounddevice 直出声卡）│
                                     └─────────────────────────────┘
训练侧（与推理错峰，同 GPU 不可并行 batch8）:
  run_preprocess.py(11步) → train_fine_tune.py(链路A) → train/train_audio2mouth.py(链路B)
      → ckpt 定版 → 手动部署 <角色>/checkpoint/{new.pt, audio2lip.pt}
```

### 2.2 端口/进程速查表

| 服务 | 进程/入口 | 端口 | 绑定 | 鉴权现状 | 解释器 | 启动脚本 |
|---|---|---|---|---|---|---|
| AI-Vtuber 中枢 | `main.py` / `webui.py` | 8086 (webui) | 本机 | 无（存量行为） | Miniconda3 py3.10 | `D:\AI\start\start_aivtuber.bat` |
| realtime_serve | `python -m realtime_serve.main` | **8000** | 127.0.0.1（默认回环） | 无 token（可选 `--api-token` Bearer 层已存在） | `D:\AI\EDTalk\runtime312\python.exe` | `D:\AI\start\start_edtalk.bat` |
| fastrag server | `server.exe`（无参数） | **11420** | **0.0.0.0（硬编码，改不了）** | **无，CORS 全开** | 无（Rust 静态二进制） | 无（需新建，见 §4.5） |
| GPT-SoVITS | `api_v2.py` | 9880 | 本机 | 无（存量行为） | runtime312 | `D:\AI\start\start_gptsovits.bat` |

> **已知接受风险（Eng S1）**：整合后 8000/11420 从"临时调试服务"变为"直播期间常驻监听"。realtime_serve 绑回环风险低；fastrag 绑 0.0.0.0 且无鉴权 = 局域网内任何人可查询/清缓存。缓解（按优先级）：① Windows 防火墙入站规则限 127.0.0.1 访问 11420；② 改 Rust 代码绑 127.0.0.1（一行改动，入 fastrag 仓库 TODO）；③ 接受（家庭可信局域网）。**当前默认 = ①+接受。**

**环境前置检查（Eng E4，启动前必过）**：
- 端口占用：`netstat -ano | findstr "8000 11420 9880"` 无 LISTEN 才可启动（占用时先杀残留进程）。
- 虚拟摄像头：UnityCapture 滤镜已注册（`UnityCapture-master\Install\Install.bat` 管理员跑过一次）；OBS 摄像头源列表可见设备。
- fastrag 索引存在：`fastrag\kbindex\chunk_index.bin` 存在（缺失时 server exit 1，需先跑 build-index.exe）。

### 2.3 资产边界速查表

| 资产 | 位置 | git 状态 | 整合处置建议 |
|---|---|---|---|
| EDTalk 代码+文档 | `D:\AI\EDTalk`（除下述排除项） | git 仓库 | submodule 候选主体 |
| runtime312 环境 | `D:\AI\EDTalk\runtime312` | 不进 git（.gitignore） | **不可搬迁**（conda 绝对路径），机器本地资产；禁止 git clean 触碰 |
| 模型权重 | `ckpts/`、`ckpt_models/`、`realtime_serve/weights/`、`wy/checkpoint/` | .gitignore `*pt`/`*.pth` 排除 | 独立分发，不随 submodule |
| wy/ 特征 npy（lip/pose 等） | `wy/**.npy` | **被 git 跟踪**（1071 变更噪音来源） | 待收敛（T4）：特征产物建议移出 git |
| fastrag 代码+配置 | `D:\AI\EDTalk\fastrag` | **非 git 仓库**，被 EDTalk .gitignore:54 整体忽略 | submodule 前需 `git init`；`chekpoint/`(4.7GB)、`kbindex/`、`target/`、`*.log` 排除 |
| fastrag 预编译 exe | `fastrag\server.exe` 等 3 个（2026-07-24 构建，~53MB） | 未跟踪 | **供应链指纹（Eng S3，待补）**：入库前记录 SHA-256（`Get-FileHash server.exe`）与构建来源（本机 cargo build --release）；后续重建 exe 时指纹比对防替换 |
| fastrag 模型 | `fastrag/chekpoint/`（MiniCPM5-1B / ritrieve_zh_v1 / mxbai-rerank-base-v2） | 未跟踪 | 独立分发；`Qwen3.5-0.8B`(1.7GB) 配置未引用，**可删除** |
| 知识库 | `fastrag/knowledgebase/`(17 md) + `kbindex/`(可重建) | 未跟踪 | 代码库收 md（内容资产），索引重建 |
| AI-Vtuber 对接蓝图 | `D:\AI\AI-Vtuber\specs\integration\*.md` | AI-Vtuber 仓库 | 已存在，A1-A4 清单直接可用 |

---

## 3. realtime_serve 实时推理服务

> 方法面与 HTTP 契约、错误包络、行为语义（切换/幂等/热切矩阵/音画同步）、三拓扑矩阵：**一律以 `realtime_serve/EMBEDDING.md` v0.3 为准**。本节只写整合实施需要的补充事实。

### 3.1 启动方式

```powershell
cd D:\AI\EDTalk    # 必须：全部内部导入为 from realtime_serve.config import ... 绝对导入
D:\AI\EDTalk\runtime312\python.exe -m realtime_serve.main --character-dir wy --target-fps 25 --face-repair on --gaze-correction on --segments auto
```

- `python -m realtime_serve` **不可用**（无 `__main__.py`），入口是模块路径 `realtime_serve.main`（main.py:513）。
- 生产实证启动命令见 `realtime_serve/data.log` 首行。
- 启动耗时以模型加载为主（644MB Generator + CUDA Graph 捕获），量级 **10s+**；`RealtimeService.__init__` 构造即全量装配（service.py:66-136）。

### 3.2 CLI 参数表（main.py:69-206，全量）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--character-dir` | `test_data/wy` | character 数据包目录（结构见 §3.5） |
| `--target-fps` | 25 | 仅约束虚拟摄像头输出节奏，不限制渲染生产速度 |
| `--cap-num` | 0 | ⚠ **dead 参数**：解析后未传入 sender（main.py:362-367 断链），多摄像头选设备不可用 |
| `--no-unitycapture` | false | 禁用虚拟摄像头（输出节奏照跑，帧丢弃计数） |
| `--batch-size` | 100 | 素材预载批大小（滑窗=2×batch，100 帧≈4s 内容） |
| `--face-repair` | None（off） | `on`（每帧~9ms GPU）\| `adaptive`（缓冲≥6帧才修，推荐） |
| `--no-face-repair` | false | off 别名（优先级高于 --face-repair，main.py:429-434） |
| `--gaze-correction` | off | `off\|on\|adaptive`；裸 flag=on |
| `--gaze-convergence` | 0.05 | 收敛角 [-1,1]，负值发散 |
| `--gaze-ckpt` | `realtime_serve/weights/current_model.ckpt` | gaze 权重（不存在则 fail-fast） |
| `--gaze-eye-source` | ori | v0.3：ori=原生眼源；rendered=legacy 回滚 |
| `--segments` | None | `auto`=全部段自然排序循环；或段名列表；启动即开播 |
| `--gen-fp16` | false | 默认关：4080S 实测慢于 fp32+TF32 |
| `--buffer-frames` | 16 | 输出环形缓冲（1..300），每帧 ~11MB @1440×2560 |
| `--preroll-frames` | 8 | 开闸前蓄帧数（≤ buffer-frames） |
| `--host` | 127.0.0.1 | API 绑定；跨机需显式 `0.0.0.0` |
| `--port` | 8000 | API 端口 |
| `--api-token` | None | 非空则全路由要求 Bearer |

Config 来源优先级：**CLI > dataclass 默认**，仅两层，无环境变量层；运行时 API 参数覆盖 CLI 值但仅存活至重启（第三层，见 EMBEDDING.md）。

### 3.3 HTTP 端点（对照 EMBEDDING.md v0.3 的差异）

端点全集与请求/响应 schema 见 EMBEDDING.md §HTTP 端点规格。本次代码核对发现 7 处差异（建议按 T3 回写契约）：

| # | 差异 | 等级 | 整合影响 |
|---|---|---|---|
| 1 | **`/audio/push` 流式路径无消费者（死路径）**：render_loop.py 全文无 audio_buffer 消费调用，PIPELINE_MAP.md §5 自标"遗留死路径" | **行为级** | **推流式音频不会产生口型**，数据只堆积内存。AI-Vtuber 必须只用 `/audio/push_full`；若 TTS 流式输出需攒整句再推 |
| 2 | `AudioPushRequest` 多 `timestamp_ms` 字段（app.py:33，被忽略） | 文档级 | 无 |
| 3 | `/segment/schedule` 响应多 `current_segment` 键（service.py:234） | 文档级（additive） | 对接代码对未知键宽容即可 |
| 4 | 401（Bearer 失败）未列入契约状态码分层 | 文档级 | 鉴权启用时需处理 401 |
| 5 | `/status` 的 `pending_switch`/`prefetch` 为条件性字段（挂起时才出现） | 文档级 | 前端按缺省处理 |
| 6 | `--cap-num` 不生效 | 实现缺陷 | 多摄像头场景暂不可选设备 |
| 7 | `Config.face_repair_enabled` 与 mode 双轨，装配只看 mode（main.py:313） | 语义陷阱 | 进程内直调时只设 `face_repair_mode` |
| 8 | **`/audio/push_full` 过载背压语义未定义**：并发/密集推送经 `_push_lock` 串行排队（render_loop.py:455），队列无界；EMBEDDING.md 只写 400/413，未写过载行为（丢弃？熔断？上限？） | **行为级（当前未定义）** | 弹幕高峰 + TTS 涌入场景下 AI-Vtuber 必须自行限流（如单飞行句 + 队列上限）；对接前按"无背压保护"设计，勿依赖服务端丢弃语义 |

> **回写优先级**：差异 #1（/audio/push 死路径）与 #8（背压未定义）为 **P1 高危回写项**（与 §10 静默失败表同级），其余 6 项为 P2 文档级回写。整合实施者请勿按"文档级"降权处理 #1/#8。

### 3.4 音频数据面（对接方必读）

- 规格：**16kHz 单声道 int16 PCM**，HTTP 侧 base64，单请求 ≤10MB（≈327s）。GPT-SoVITS 输出 16kHz 已确认（对接纪要 §5），**无需转码**。
- 首选 `POST /audio/push_full {pcm_base64}`：全序列异步推理，唇形最稳；每次 push 自动 flush+preroll，句间自然衔接。
- 音画同步 = 消费端帧锁定：口型领先仅 DAC 延迟 ≤~40ms 绝不滞后；推理未就绪时音频等待（HOLD）。
- 服务经 sounddevice 直出声卡。**若 AI-Vtuber 另有统一音频播放管线，需决策由谁出声**（双路同时出声会重复）。

### 3.5 character 数据包结构

```
<character_dir>/            # 例：wy/（= 预处理 base_dir 同一目录）
├── orifra/<段名>/N.jpg     ★必需 全帧 JPEG（输出底帧/贴回目标；段名=orifra 子目录名）
├── count/<段名>/N.jpg      ★必需 256² crop JPEG（Generator 唯一输入）
├── metadata/<段名>.npy     ★必需 裁切坐标（贴回）
├── facemark/<段名>.npy     ★必需 (N,68,2) 256 空间 68 点（mask + gaze 回退）
├── gaze_data/<段名>.npy    ◆强烈建议 眨眼标签/眼中心查表（缺席→实时回退，噪声大）
└── checkpoint/
    ├── audio2lip.pt        ★必需 ~11MB，含 'audio2lip' 键（链路B 产物）
    └── new.pt              ★必需 ~644MB，含 'gen' 键（链路A 产物或底模）
```

- 段名区分大小写、支持中文（wy 实测：1/11/22/33/44/55）。
- 段内 orifra/count/metadata/facemark 四处条数必须严格一致（sequence_frame_pack.py:962-967 校验，不一致 ValueError）。
- 训练专用目录（lmdb/mel/lip_feature/…）服务**完全不读**，留在包内无碍；精简部署包只需上列 5 目录 + checkpoint。
- 混合分辨率：各段原生直出，段切换时摄像头按新尺寸重建（消费端需容忍短暂 letterbox/重连）。

### 3.6 模型权重清单

| 文件 | 大小 | 加载者 | 时机 |
|---|---|---|---|
| `<角色>/checkpoint/audio2lip.pt` | 11.3MB | ModelManager（models.py:104-115） | 启动必载 |
| `<角色>/checkpoint/new.pt` | 644MB | ModelManager Generator（models.py:118-129） | 启动必载 + CUDA Graph 捕获 |
| `realtime_serve/weights/current_model.ckpt` | 228MB | GazeCorrector | gaze≠off 才载；缺失 fail-fast |
| `realtime_serve/weights/facefix256.onnx` | 76MB | FaceRepair (GPEN-256, ORT CUDA) | repair≠off 才载 |

无懒加载：启动 off 的 gaze/repair 运行时热开端点返回 409（`{mode}_capable=false` 先探测）。

### 3.7 进程生命周期与运维

- **stop 不可逆**：stop 后 start 抛 RuntimeError，重启=新进程（service.py:144-145）。
- **无自恢复**：渲染线程连续 ~30s 失败熔断自杀后，HTTP 面仍在（`/status.last_error`）但画面停摆——**AI-Vtuber 侧必须做 supervisor**：拉起（cwd=项目根 + runtime312 解释器）→ 轮询 `/status` 到 `preroll_state=="complete"` 且 `current_segment` 非空 → 崩溃重启。
- **日志纯 stdout**：全仓无文件日志代码，data.log/log16.txt 是人工重定向快照。整合时由 AI-Vtuber 重定向 stdout 到文件并管理轮转。渲染线程每 5s 打 FPS/buffer/p99 摘要（render_loop.py:440-449）。
- **GPU 占用**：常驻 3-4GB 量级（Generator+CUDA Graph、Audio2Lip、Gaze 228MB、GPEN ORT）。**与训练并行冲突**：训练 batch8 峰值 15.05GB/16GB 必然溢出，batch4 处于临界——训练与推理必须错峰（§8.4）。

### 3.8 虚拟摄像头

- pyvirtualcam 0.15.0，`backend='unitycapture'`，与本仓 DH-Live 定制 UnityCapture DirectShow 滤镜经共享内存通信。
- 滤镜注册：`D:\AI\EDTalk\UnityCapture-master\Install\Install.bat`（**需管理员运行一次**）；多设备用 InstallMultipleDevices.bat。
- 回退陷阱：backend 失败按 `['obs','unitycapture',None]` 依序尝试——**机器若注册过 OBS 虚拟摄像头而未装 UnityCapture，会静默走 OBS 后端**，排查 `_active_backend`。
- 初始化成功即发一张绿色 "EDTalk Ready" 测试帧（unity_capture.py:81-86）——AI-Vtuber 启动后看到的第一帧。
- 就绪后输出分辨率=首段 orifra 原生尺寸；underrun 重发上一帧保喂帧。

---

## 4. fastrag 知识库问答服务

### 4.1 部署形态与启动

- 三个 exe（`server.exe` 常驻 / `query.exe` 交互 CLI / `build-index.exe` 一次性）**均无命令行参数**（grep 无 clap/env::args），一切由 `Config.toml` + 环境变量驱动。
- **Config.toml 三级查找**（config.rs:157-186）：① `FASTRAG_CONFIG` 环境变量（指向具体文件）→ ② 当前工作目录 `./Config.toml` → ③ exe 所在目录。**全部相对路径（模型/知识库/索引）基于 Config.toml 所在目录解析，与 CWD 无关**。
- **整合推荐启动方式**：设 `FASTRAG_CONFIG` 绝对路径，CWD 任意。陷阱：AI-Vtuber 若以自己目录为 CWD 启动而该目录恰有 Config.toml 会错拿（优先级②高于③）。

```powershell
# AI-Vtuber 侧拉起（推荐）
$env:FASTRAG_CONFIG = "D:\AI\EDTalk\fastrag\Config.toml"
Start-Process -FilePath "D:\AI\EDTalk\fastrag\server.exe" -WindowStyle Hidden `
  -RedirectStandardOutput "D:\AI\logs\fastrag_stdout.log" -RedirectStandardError "D:\AI\logs\fastrag_stderr.log"
# 就绪探活：轮询 GET /health 直到 200（启动 ~2.5-4s：Embedder CPU 1.5s + LLM CUDA 1.2s + Reranker 0.6s）
```

- 退出码：`0`=正常完成（含 build-index 的"知识库目录不存在自动建空目录后退出"分支——**注意这不是错误**）；`1`=仅 server 索引缺失；`101`=任意 panic（配置缺失/模型加载失败/端口占用）。

### 4.2 配置体系（Config.toml 全节 + 环境变量）

| Section | 关键项 | 当前值 | 说明 |
|---|---|---|---|
| paths | model_dir / kb_dir / index_dir | chekpoint / knowledgebase / kbindex | 相对 Config.toml 所在目录 |
| retrieval | enable_rerank / top_k_candidates / rerank_top_k / max_chunk_chars | true / 6 / 2 / 512 | rerank 失败自动降级为向量结果 |
| quantization | embedder_device / embedder_dtype | cpu / fp32 | 仅 Embedder 可配；LLM/Reranker 自动 CUDA+BF16 |
| llm / embedder / reranker | model_dir | MiniCPM5-1B / ritrieve_zh_v1 / mxbai-rerank-base-v2 | 三模型目录必需（§4.6） |
| generation | max_answer_length / max_new_tokens / temperature / top_p / do_sample / enable_thinking / repeat_penalty / top_k_sampling | 80 / 128 / 0.25 / 0.9 / true / false / 1.1 / 40 | 答案 80 字硬截断 |
| server | port / max_batch_size | 11420 / **1** | max_batch_size=1 时批量接口静默截断 |
| prompt | system_role / answer_rules | "老王"主播人设 + 回答规则 | 人设与答案规则在此改 |

**环境变量覆盖（实测勘误）**：真实写法是 `FASTRAG_<SECTION>__<KEY>`，即前缀与 section 之间**单个 `_`**、section 与 key 之间**双 `__`**：

```powershell
$env:FASTRAG_SERVER__PORT = "11421"           # ✅ 生效（实测）
$env:FASTRAG__SERVER__PORT = "11421"          # ❌ 不生效（Config.toml:3 与 README.md:108 的注释是错的）
```

拼错静默忽略无提示；排错时可用设非法值（如 `FASTRAG_SERVER__PORT=abc` → panic exit 101）验证覆盖是否生效。范围覆盖全部配置项。

### 4.3 HTTP API 表（端口 11420）

| 端点 | 方法 | 请求 | 成功响应 | 错误 |
|---|---|---|---|---|
| `/health` | GET | — | `{"status":"ok","chunks":145,"rerank_enabled":true,...}` | — |
| `/query` | POST | `{"question":"..."}`（非空，≤2000 字符） | `{"question","answer","context":[{content,source,doc_title}],"mode":"fastrag"}` | 400/500，**错误体为纯文本非 JSON** |
| `/batch_query` | POST | `["问题1","问题2",...]`（字符串数组） | `{"results":[{"question","answer"}]}` | 单条失败 answer="错误: ..."；**超 max_batch_size 静默截断（当前=1，只处理第一条）** |
| `/clear_cache` | POST | — | `{"status":"ok",...}` | — |

- **无鉴权 + CORS 全开 + 硬编码绑定 0.0.0.0**（server.rs:183-186,199；端口可 env 改，绑定地址改不了需改代码）。缓解：Windows 防火墙入站规则限 127.0.0.1，或改代码。
- **串行推理**：`inference_lock` 全局锁，并发请求排队；单条实测 ~2.53s（eval_results.json）。弹幕高峰需前端做超时丢弃或队列上限，**逐条调用 /query，不要用 /batch_query**。
- **客户端模板**：`fastrag/api_client.py` 的 `LiveStreamBot` 类（api_client.py:81-124）即官方集成样板：3 秒探活 + 单条 query + 三档异常降级话术（超时/离线/兜底），保证直播不冷场。

### 4.4 知识库构建流程

```
knowledgebase/ (*.md, *.txt, 递归, UTF-8)
  → build-index.exe（仅加载 Embedder，秒~分钟级）
      Markdown 标题分块：# H1=文档名(doc_title)；## H2=主分块边界；### H3=细分
      每块 content 首行 = 标题路径纯文本拼接（为中文向量检索设计）
      硬约束：标题行必须 "# 标题"（# 后有空格），"##标题" 不识别
  → kbindex/{chunk_index.bin, data.json}   （当前：17 md → 145 块 × 1792 维）
  → 重启 server.exe（**索引只在启动时加载一次**，运行中重建不被感知）
```

- **全量重建、非增量**：无文档级哈希/时间戳机制；145 块规模下全量重建很快，每次直接重跑即可。
- 写作规范见 `fastrag/知识库规范.md`（每块 30~300 字、同义词/口语变体覆盖等 7 条）。

### 4.5 无控制台运行评估

- 三个 exe 均为**控制台子系统**（Cargo.toml 无 `windows_subsystem` 属性）：直启必弹控制台窗口。
- 日志：tracing → **stdout**，panic → **stderr**；目录里的 server.log/query.log 是人工捕获的终端记录，**不是程序写出的**；无文件日志、无轮转代码。
- 方案对比：
  - **A（推荐，零改造）**：`Start-Process -WindowStyle Hidden` + stdout/stderr 重定向到文件（§4.1 命令）。窗口短暂创建后隐藏，可接受；日志由外部管理轮转。
  - B（改造）：加 `#![windows_subsystem = "windows"]` 彻底无窗口，**但 stdout 日志全丢**——必须同时补 tracing-appender 文件日志，属代码改动，入 TODOS。
- 停机：server 无 ctrl_c 优雅停机处理，AI-Vtuber 停止时 `taskkill /F /T`（与现有 coordination_program 停止方式一致）。
- **崩溃取证（Eng H3）**：无控制台/Hidden 窗口模式下崩溃无可见痕迹——stdout/stderr 重定向文件就是唯一取证面。检查顺序：① 重定向文件尾部的 panic 行（`thread 'main' panicked at src\...`）+ 退出码 101=模型/配置加载失败；② 退出码 1=索引缺失（先跑 build-index.exe）；③ 无输出但进程消失=系统级杀死（查 AI-Vtuber 侧 taskkill 记录）。日志无轮转，AI-Vtuber supervisor 按日期切文件。

### 4.6 模型目录与 git 现状

- `chekpoint/` 需要三个子目录：`MiniCPM5-1B`（LLM）、`ritrieve_zh_v1`（Embedder，含 2_Dense 投影则 1792 维）、`mxbai-rerank-base-v2`（Reranker，enable_rerank=true 时必需，缺失降级 warn 继续起）。模型文件缺失 = 启动 panic（exit 101），不降级。
- `Qwen3.5-0.8B`（1.7GB）：配置未引用、代码从不加载，**搬运/入库时排除**。
- git 现状：fastrag **无 .git**，被 `D:\AI\EDTalk\.gitignore:54` 整体忽略。submodule 化步骤：fastrag 内 `git init` → .gitignore 排除 `chekpoint/`、`kbindex/`、`target/`、`*.log`、`fastrag-python/`（旧 Python 参考实现，含嵌套 .git，整个排除）、`pyproject.toml`（HKU 无关误留文件）→ 首提交 → AI-Vtuber 侧 `git submodule add`。模型走独立分发。
- 显存参考：LLM BF16 ~2.1GB + Reranker ~1GB；Embedder 留 CPU。与 realtime_serve 同机并存显存安全（合计 <7GB / 16GB）。

### 4.7 前端指令集成对照表（AI-Vtuber 侧）

AI-Vtuber 指令体系 = `utils/handlers/comment_handler.py:240` `comment_handle()` 串行责任链（依次短路，返回 True 即消费）。fastrag 两个挂载点：

| 挂载点 | 位置 | 模式 | 实现 |
|---|---|---|---|
| **知识库指令**（弹幕触发问答） | comment_handle 责任链新插一环（建议在"自定义命令"之后、"本地问答"之前） | 照 `choose_song_handle` 模式（feature_handler.py:207-209）：config.enable 开关 + `starts_with_any` 前缀匹配 + 命中后调 `/query` + 返回 True 短路 | 新建 `utils/edtalk_realtime/fastrag_client.py`（照 AUDIO_PLAYER 模式：requests + api_ip_port + try/except + logger + LiveStreamBot 降级话术） |
| **LLM 增强挂载** | `config.chat_type` → `my_handle.py:190` GPT_MODEL.set_model_config（注册表在 utils/gpt_model/gpt.py） | 把 fastrag 的 `/query` 包装为一种 chat_type，或作为 LLM 前置检索（local_qa 相似度匹配的向量升级） | 二选一：指令型（上排）为主，chat_type 型为辅 |
| 外部 HTTP 指令入口 | `utils/web_server.py:77` `/send`（type=comment 分发） | 非弹幕来源（面板手动触发）走此口 | 无需新代码 |

配置节：`config.fastrag`（照 metahuman_stream 节模式：`api_ip_port` 默认 `http://127.0.0.1:11420`、`enable`、指令前缀、超时、降级话术开关）+ `webui.show_card` 显隐 + migrate_config/split_config 迁移。

> **信任边界（Eng S2）**：弹幕是**不可信输入**。挂载进 comment_handle 责任链后，观众消息将触达 fastrag 检索与 LLM 生成并直接进直播输出——指令前缀必须可配置且默认关闭，上线前先在"仅主播/面板触发"模式验证；知识库内容经由 LLM 的注入面为存量行为，指令链不扩大它，但**触发者 = 观众**这一点必须在配置评审时知情确认。

---

## 5. 数据预处理管线（11 步）

入口：`python data_preprocess/data_preprocess_for_train/run_preprocess.py --base_dir <dir> --safe_mode`（Windows 首选）。**唯一原始输入 = `<base_dir>/original_videos/`**（启动即校验）。子进程串行执行，`-u` 无缓冲 + UTF-8 env 注入，失败自动重试 `--max_retries`(5) 次，各步幂等实现断点续跑。

### 5.1 步骤明细表

| 序 | 脚本 | 输入 | 输出（目录名固定） | 关键透传参数 | 幂等性 |
|---|---|---|---|---|---|
| 1 | crop_video_HDTF.py | original_videos/ | video/ | image_shape 256,256、detect_interval、detect_timeout_s、num_processes(Win=1) | ✅ 按视频跳过 |
| 2 | split_HDTF_video.py | video/ | split_5s_video/ | duration 5 | ❌ 全量重切重编码（重跑代价高） |
| 3 | split_HDTF_audio.py | original_videos/ | audios/ + split_5s_audio/ | duration 5 | ◐ 整段 wav 跳过；切分不跳过 |
| 4 | prepare_lmdb.py | split_5s_video/ | lmdb/ | n_worker | ❌ 全量重建 |
| 5 | get_mel.py | split_5s_audio/ | mel/ | 无 | ✅ 按 npy 跳过（⚠ 内部硬编码 .cuda()，无 CPU 回退） |
| 6 | extract_lmdk.py | lmdb/ | landmark/ | num_processes(Win=1) | ✅ 主级过滤（纯 CPU dlib） |
| 7 | extract_bbox.py | lmdb/ | bbox/ | num_workers(Win=1) | ✅ 双重过滤（GPU S3FD，每 worker 独占一份） |
| 8 | extract_lip_pose_features.py | lmdb/ + **ckpts/EDTalk_lip_pose.pt** | lip_feature/(N,20)、pose_feature/(N,6) | gpu_id | ✅ 双文件过滤，支持 --force |
| 9 | prepare_feature_lmdb.py | mel(基准)+lip/pose/bbox | lmdb_features/ | （无透传） | ◐ 重跑覆盖；**--force 删库重建** |
| 10 | crop_head_video.py | original_videos/ | orifra/、count/、metadata/、facemark/ | device、num_workers(线程)、output_size 256、output_format、longest_edge 1920、no-landmark(safe_mode) | ◐ 拆帧/裁帧断点续传；**metadata/facemark 仅末尾一次写入（见 5.3 地雷）** |
| 11 | prepare_gaze_data.py | facemark/ | gaze_data/<段>.npy | （无透传） | ❌ 无条件重算覆盖（纯 CPU 秒级） |

### 5.2 Windows 注意点

- **spawn Pool 假死**（最大坑）：步骤 1/6/7 在 Py3.12 下 spawn Pool 有概率 worker 崩溃永久假死——run_preprocess **已强制钳为 1 进程**（无论 safe_mode）。线程池步骤（4/10）可放心高 num_workers；步骤 10 的 num_workers 传原值不钳 1。
- `--safe_mode` = num_workers 1 + detect_interval≥3 + no_landmark（步骤 10 无关键点 → gaze_data 不可用 → gaze 回退 facemark 实时计算）。
- `--detect_interval` 只覆盖步骤 1；步骤 10 恒每帧检测（CLI 默认，未透传）。
- ffmpeg：步骤 1/10 直接调 `runtime312/ffmpeg/bin/ffmpeg.exe`；步骤 2/3 前置该目录进 PATH。
- GPU 分布：1=GPU+CPU、5=**硬编码 .cuda()**、7=GPU、8=GPU 串行、10=GPU+CPU；其余纯 CPU。

### 5.3 步骤 10 中断地雷（高危）

metadata/facemark 只累计**本次运行**处理的帧。若视频中途中断后续跑：count/ 帧数 > metadata 行数 → realtime_serve 加载时 `ValueError`（四目录条数强校验）。**中断后该视频应删净 orifra/count/metadata/facemark 四处重跑**（facemark 行数=帧数还有 fail-fast：fine_tune/dataset.py:113-116）。

### 5.4 根目录 crop_head_video.py 是弃用旧版

`D:\AI\EDTalk\crop_head_video.py`（7 月 30 日）与 `data_preprocess/data_preprocess_for_train/crop_head_video.py`（8 月 18 日）为分叉的两个版本：旧版 cv2 拆帧、pad_ratio 1.05-1.2；新版独立 ffmpeg 拆帧+断点续传、pad_ratio 1.2-1.3。run_preprocess 永远取新版（resolve_script 优先 SCRIPT_DIR）。**手工运行请以 data_preprocess_for_train 版为准**；根目录版建议删除（防误用产生几何不一致的 count/facemark）。

### 5.5 产出 → 消费方映射

| 产出 | 链路A fine_tune | 链路B audio2mouth | realtime_serve | 备注 |
|---|---|---|---|---|
| video/、audios/、split_5s_*、landmark/ | — | — | — | 中间产物；**landmark/ 当前主链无消费方**（旧解耦训练遗留） |
| lmdb/ | — | **必需** | — | 图像源+视频名册 |
| mel/ | — | **必需** | — | 每帧 (80,16) 200ms 窗 |
| bbox/ | — | **必需** | — | sync/嘴部 VGG crop |
| lip_feature/、pose_feature/ | — | **必需**（学习目标） | — | 由底模 EDTalk_lip_pose.pt 提取（§9 模型链耦合） |
| lmdb_features/ | — | 可选加速 | — | 存在即用，否则回退 npy |
| count/、facemark/ | **必需** | — | count 必需 / facemark 必需 | facemark 行数=帧数强校验 |
| orifra/、metadata/、gaze_data/ | — | — | **必需/必需/强烈建议** | gaze_data 缺席→实时回退噪声大 |
| checkpoint/{audio2lip,new}.pt | — | — | **必需** | 训练产物部署（§9） |

---

## 6. 链路A：底模微调 train_fine_tune.py

**训练对象**：Generator(lip_pose 版)，主模式只训 dec。**底模**：`ckpts/EDTalk_lip_pose.pt`（键 `{gen,dis,g_optim,d_optim,args}`）。

### 6.1 数据依赖与运行

```powershell
cd D:\AI\EDTalk    # exp_path/resume_ckpt 均相对 cwd，必须项目根运行
D:\AI\EDTalk\runtime312\python.exe train_fine_tune.py --datapath wy --exp_name wy_runC --only_fine_tune_dec --iter 50000
```

- 数据：`<datapath>/count/<视频名>/*.jpg`（数字序）+ `<datapath>/facemark/<视频名>.npy`（缺省解析 datapath 同级 facemark/，`--facemark-dir` 可覆盖；行数≠帧数直接 RuntimeError）。
- 多视频（≥2）默认跨视频采样（防单人捷径）；MAR 张嘴池 top 25% 过采样 ratio 0.5（牙齿学习信号）；每视频末 5% 帧剔除（held_out，评估须同口径）。

### 6.2 关键参数（完整 argparse 见脚本；此处列整合需显式下发的）

| 参数 | 默认 | 整合建议 |
|---|---|---|
| `--only_fine_tune_dec` | off | **必须显式开启**：冻结 enc/fc/direction，保住与 Audio2Lip 输出的接口一致性（§9 模型链耦合的解除条件） |
| `--iter` | 800000 | 实战 50000（~3.5h） |
| `--batch_size` | **8** | ⚠ help 写"默认 4"但 default=8；8=15.05GB 贴 WDDM 悬崖，**建议显式传 4**（12.03GB 安全区） |
| `--lr` | **0.001** | ⚠ help 写"0.0001"但 default=0.001；微调建议显式传更小值按配方 |
| `--resume_ckpt` | ckpts/EDTalk_lip_pose.pt | 底模起点 |
| `--exp_path/--exp_name` | ckpt_models / Obama | 产物位置（§9 流转表） |
| `--save_freq` | 5000 | 崩溃保护间隔 |
| `--held-out-ratio` | 0.05 | 与 tools/eval_finetune.py 口径对齐 |

（注意 `--held-out-ratio` 连字符参数名，argparse 自动转下划线。）

### 6.3 产物与 Windows 防护

- 产物：`ckpt_models/<exp_name>/checkpoint/<iter:06d>.pt`，键 `{gen,dis,g_optim,d_optim,args}`——与底模文件格式**同构**。
- 崩溃保护：KeyboardInterrupt/异常 finally 紧急保存，不丢进度。
- 防护三件套自动继承：spawn 强制、cudnn.benchmark=False（RTX40+cu128 kernel bug）、WDDM 定期 sync（`--wddm_sync_interval 500`）。
- 定版：`python tools/eval_finetune.py --cand-ckpt <产物>`（门槛见 RUNBOOK.md:211-216：姿势 MAE ≤11.1px、开合比 0.8-1.25 等）。

---

## 7. 链路B：Audio2Lip 微调 train/train_audio2mouth.py

**训练对象**：Audio2Lip（mel 200ms 窗 → 20 维 lip feature）。**底模**：`ckpts/Audio2Lip.pt`（键 `{audio2lip,args}`）+ Audio2Lip 构造时自动吃 `ckpts/Wav2Lip.pt` 预加载编码器。

### 7.1 数据依赖与运行

```powershell
cd D:\AI\EDTalk    # 只能 python train/train_audio2mouth.py（不能进 train/ 目录跑：
                   # sys.path 双重布局——train/ 提供 vgg19/networks_Lip_NonLip，项目根提供 datasets/util）
D:\AI\EDTalk\runtime312\python.exe train\train_audio2mouth.py --preload_features --data_path wy `
  --resume_ckpt ckpt_models/wy_runC/checkpoint/035000.pt `
  --audio2lip_ckpt ckpts/Audio2Lip.pt `
  --epoch 2 --batch_size 4 --lr_schedule 0.001,0.001 `
  --lip_dim_weight 5.0 --mouth_vgg_weight 0.2 --vgg_weight 0.2 `
  --vgg_sparse_freq 50 --smooth_weight 0.2 `
  --audio_encoder_unfreeze_from 9 --audio_encoder_lr_mult 0.1 --amp --exp_name wy
```

（= wy 角色定稿配方，见 §9.2；无 SyncNet——单人数据消融实证无贡献，`--sync_weight` 默认关。）

- 数据：`lmdb/` **必需**（缺即 FileNotFoundError）+ `mel/lip_feature/pose_feature/bbox` .npy + `lmdb_features/` 可选加速。
- `--preload_features`：全量特征入内存（63055 clips 数据集实测可用），小数据集推荐。

### 7.2 双 resume 语义（易混淆，编排必须区分）

| 参数 | 语义 | 格式要求 |
|---|---|---|
| `--resume_ckpt` | 训练主 checkpoint：新格式（链路B 自产，含 `audio2lip` 键）只恢复训练状态；**旧格式（链路A 产物，含 `gen` 键）会把 gen 加载进训练**——梯度经该 gen 反传，需与部署组合一致 | 文件名必须 `NNNNNN.pt` 数字格式（start_iter 从文件名解析） |
| `--audio2lip_ckpt` | 独立的 Audio2Lip 起点权重（官方底模），仅在 resume_ckpt 无 `audio2lip` 键时回退加载 | `{audio2lip,args}` |

### 7.3 产物与硬约束

- 产物：`logs/<exp_path>/<exp_name>/<date>/checkpoint/<iter:06d>.pt` 与 `epoch_%03d/<iter:06d>.pt` 子目录；键 **只存 `{audio2lip, args, start_iter}`**——gen/dec 有意不存（微调 dec 实测不如底模，用户两次独立验证后正式关闭），**编排器不要"顺手"补存 dec**。
- DataLoader **强制 num_workers=0**（Windows spawn worker 的 LMDB mmap 与 WDDM CUDA 页表冲突 → C 级 access violation）；`--num_workers` 参数形同虚设。
- empty_cache 全面禁用（触发 nvlddmkm.sys BSOD），只 gc.collect；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（max_split_size_mb/roundup_power2_divisions 在 WDDM 下 segfault，勿加）。
- SIGINT/SIGTERM 优雅关闭（保存后退出）；异常紧急保存后 re-raise（编排以退出码判失败）。
- 时长参考：~1.5 it/s，4 epochs ≈ 11.8h（batch4）；定版配方 2 epochs ≈ 6h 量级。

---

## 8. 一键训练编排蓝图

> 本节是"一键角色适配"（TODOS P2）的前置设计。编排脚本属实施阶段；实施时按本节顺序与约束固化。

### 8.1 编排顺序（有实战依据）

```
[输入] 角色视频 → <base_dir>/original_videos/

第1段 预处理（GPU+CPU 混合，小时级）
  run_preprocess.py --base_dir <base_dir> --safe_mode
  ▸ 全新角色跑全 11 步；仅换训练配方重训时按 §8.3 最短链补跑

第2段 链路A 底模微调（独占 GPU ~3.5h）
  train_fine_tune.py --datapath <base_dir> --exp_name <角色>_runN --only_fine_tune_dec --iter 50000
  ▸ 验收门：tools/eval_finetune.py --cand-ckpt <产物>（门槛 RUNBOOK.md:211-216）
  ▸ 不过门 → 调配方重训或终止（劣化模型不上线）

第3段 链路B Audio2Lip 微调（独占 GPU ~6h/2ep）
  train_audio2mouth.py --data_path <base_dir> --resume_ckpt <A 定版> --audio2lip_ckpt ckpts/Audio2Lip.pt ...
  ▸ 定版：eval 曲线 + best checkpoint（E2 拐点经验：epoch_001 末段）

第4段 部署打包（分钟级，手动/脚本均可）
  copy <A 定版>.pt  → <base_dir>/checkpoint/new.pt        （含 gen 键，文件名固定）
  copy <B 定版>.pt  → <base_dir>/checkpoint/audio2lip.pt   （含 audio2lip 键，文件名固定）
  ▸ base_dir 即 character_dir（wy 模式：目录两用）；实时服务重启后生效
```

### 8.2 编排器功能要点

1. 编排单位 = 角色数据目录（base_dir），三段共用；**"产物存在即跳过"**的幂等设计可行（预处理各步幂等、两链路支持 resume）。
2. 链路B 必须在链路A **定版之后**启动（resume_ckpt 指向定版 .pt），否则部署组合漂移。
3. 两链路输出目录模式不同（`ckpt_models/<exp>/checkpoint/` vs `logs/<exp_path>/<exp>/<date>/checkpoint/`），编排器需分别做产物发现与定版挑选（A 侧 eval 门槛自动判，B 侧 best/epoch 指标）。
4. 部署是"带键改名"而非整文件复制：文件名 `new.pt`/`audio2lip.pt` **固定不可改**（sequence_frame_pack.py:1162-1163 fail-fast）。
5. 失败语义：A 崩溃自动紧急保存（退出码仍非 0，可续训）；B 紧急保存后 re-raise——编排以**进程退出码**判失败，勿以"有没有新文件"判断。
6. **步骤级日志**：每段起止时间、退出码、产物路径、显存水位（`torch.cuda.memory_allocated`）、**坏样本计数**落 JSONL，供断点续跑与排障（S8 缺口闭合点）。
7. runtime312 自检脚本注意：`runtime312/test_audio2lip_direct.py`、`test_audio2lip_init.py` 引用 audio_encoder，属环境自检——清理 runtime 目录会使其失效（次要备注）。

**每步失败恢复语义三态与完成判据（Eng E2/T3）**：

| 步骤 | 失败后行为 | 完成判据（产物 + 最小规模） |
|---|---|---|
| 预处理 1-9 | 重跑安全（幂等跳过已处理）；2/3/4 重跑=全量重做（代价高但正确） | 各输出目录非空；`mel/` npy 数 = split_5s_audio 段数 |
| 预处理 10 | **禁止盲续跑**：中断即删净该视频 orifra/count/metadata/facemark 重跑（行数不一致地雷 §10-8） | 每段 count 帧数 = metadata 行数 = facemark 行数 = orifra 帧数 |
| 预处理 9（重提特征后） | **必须 `--force`**（否则旧特征混入 §10-4） | lmdb_features mtime 晚于全部特征 npy |
| 链路A | 崩溃=紧急保存，`--resume_ckpt <紧急产物>` 续训 | eval_finetune 过门槛（RUNBOOK 211-216）；产物含 gen 键 |
| 链路B | 崩溃=紧急保存后 re-raise，`--resume_ckpt <产物>` 续训（新格式不回灌 gen） | eval 曲线平台期 + best ckpt 含 audio2lip 键 |
| 部署打包 | 重跑安全（copy 幂等） | §9.3 三重验证（文件名+键+大小）通过 |

**坏输入路径可见性（Eng E3）**：预处理各步对损坏视频/检测失败帧/空 clip 的处理不一（步骤 1 有效帧不足删输出、步骤 7 只保留有检测帧、步骤 10 失败帧写 NaN 行）——编排器日志必须输出**每步的跳过/失败计数**，计数 >0 时醒目标红，防止"静默产出残缺数据集"。

### 8.3 重训场景的最短链（换配方不换数据）

记忆档案实证的换角色/换配方最短链：**步骤 4（LMDB 入库）→ 步骤 5（get_mel）→ 步骤 8（lip/pose 特征）→ 步骤 9（prepare_feature_lmdb `--force`）→ 链路B**。仅动链路A（dec 微调）时链路B 产物无需重训（§9.1 条件）。

### 8.4 GPU 错峰约束（硬性）

- 链路A（batch4 12.03GB / batch8 15.05GB）、链路B（batch4 ~5.79GB+嘴部VGG增量）、realtime_serve 常驻 3-4GB、fastrag ~3.1GB。
- 16GB 卡上：**训练（A 或 B）与 realtime_serve 推理不可并行（batch8 必然溢出，batch4 临界）**；fastrag 与推理并存安全。
- 编排器在启动训练段前应检查 realtime_serve 是否在跑（/status 探活），在跑则拒绝或提示确认。

---

## 9. 配方与 checkpoint 流转（一等公民）

> 微调配方与 checkpoint 的可复现性是本项目的护城河载体（口型质量差异化所在）。本节是配方变更的唯一登记处。

### 9.1 模型链耦合（核心约束，决定重训范围）

**数值契约（Eng H1，任一项改动 = 需重跑预处理 + 重训 + 重部署）**：

| 契约项 | 值 | 断裂现象 |
|---|---|---|
| 音频采样率 | 16kHz（训练数据链与推理 push_full 一致） | 唇形静默错位/开合错乱 |
| mel 窗口 | 200ms（hop 200 → 80 mel帧/s；每视频帧取 16 列，起点=当前帧前 2 帧） | 训练/推理 mel 语义不一致 → audio2lip 输出失效 |
| 训练 clip 长度 | T=5 帧（mel/lip/pose/bbox/target 同切 `r:r+5`） | dataset 构建 clip 起点集错误 → 训练数据残缺 |
| lip feature 维度 | 20 维（dim 2/13/16=开合度） | audio2lip 与 Direction 接口断裂 |
| pose feature 维度 | 6 维 | 同上 |
| 图像域 | 256×256 crop，[-1,1] 归一化 | 贴回/渲染错位 |


```
ckpts/EDTalk_lip_pose.pt['gen'] ──(步骤8 特征提取器)──► lip_feature 空间（20维）
                                                              │
链路B 的 audio2lip 学习目标 = 这套 lip_feature ◄──────────────┘
```

- **链路A 锁 `--only_fine_tune_dec`（enc/fc/direction 冻结）时 lip feature 空间不变** → 链路B 产物继续有效，无需重训。Run C 已验证该路径。
- 若链路A 全量微调（动了 direction）且产物替换 `ckpts/EDTalk_lip_pose.pt` → **步骤 8 必须重跑、链路B 必须重训**（lip_feature 空间已变）。`--direction_lr_ratio` help 明确警告调高会"破坏与 Audio2Lip 输出的接口一致性"。
- `ckpts/EDTalk.pt`（expression 版，键 `{gen,g_optim,args}`）与两条链的 Generator 结构**不兼容**，不可顶替 EDTalk_lip_pose.pt（extract_lip_pose_features.py:209-211 显式警告）。

### 9.2 锁定配方（wy 角色，2026-09 定稿）

| 链路 | 配方 | 时长 | 已证伪（勿回头） |
|---|---|---|---|
| A 底模微调 | `--datapath wy --exp_name wy_runC --only_fine_tune_dec --iter 50000`（v3 默认已内置：跨视频/MAR 过采样/嘴区 crop 监督/temporal 移除） | ~3.5h | temporal loss（背景失效）、全量微调动 direction（破坏接口）、mouth_weight>6 无效（特征分辨力瓶颈非权重） |
| B Audio2Lip 微调 | 见 §7.1 命令（2 epochs + lip_dim_weight 5.0 + mouth VGG 0.2 + smooth 0.2 + block9 分层解冻 + 无 SyncNet + AMP） | ~6h/2ep | epoch>2、SyncNet 监督、VGG 稀疏×N 补偿（崩坏）、lip_dim_weight>5（误差守恒）、**保存微调 dec（两次独立实测不如底模，正式关闭）**、时序上下文方案（触发条件未满足，见记忆档案） |

### 9.3 checkpoint 流转表

| 产出方 | 路径模式 | dict 键 | 消费方 |
|---|---|---|---|
| 链路A save() | `ckpt_models/<exp>/checkpoint/<iter:06d>.pt` | `{gen,dis,g_optim,d_optim,args}` | ① eval_finetune 定版 → ② `gen` 键 → 部署 `<角色>/checkpoint/new.pt`；③ 整文件替换 `ckpts/EDTalk_lip_pose.pt`（影响：步骤8 特征提取器、demo 默认、下一轮底模、快照测试 test_base_model_snapshot）；④ 链路B `--resume_ckpt`（旧格式分支） |
| 链路B save() | `logs/<exp_path>/<exp_name>/<date>/checkpoint/<iter:06d>.pt`（+ epoch_%03d/ 子目录） | `{audio2lip,args,start_iter}` | ① 定版 → `audio2lip` 键 → 部署 `<角色>/checkpoint/audio2lip.pt`；② 自己 resume（新格式） |
| 自训 SyncNet（可选） | `ckpt_models/syncnet/syncnet_best.pt` | `{epoch,state_dict,optimizer,scheduler,loss}` | `--syncnet_ckpt`；单人数据建议直接不用 |
| 底模 | `ckpts/EDTalk_lip_pose.pt` `{gen,dis,...}`；`ckpts/Audio2Lip.pt` `{audio2lip,args}`；`ckpts/Wav2Lip.pt`（Audio2Lip 构造自动吃） | | |

**部署契约（唯一硬性接口）**：`<character_dir>/checkpoint/` 下 `new.pt`（含 gen 键）+ `audio2lip.pt`（含 audio2lip 键）**同时存在、文件名固定**；加载端 `torch.load(..., weights_only=False)`（ckpt 内嵌 pickle args——只加载自产 ckpt，勿引入外部文件，RUNBOOK.md:222）。

**手动部署步骤**（当前无自动脚本，TODOS 登记；**每步附产物验证，Eng H2**）：
```powershell
# 1. 链路A 定版 → new.pt（部署前先验证产物存在且含 gen 键）
python -c "import torch; ck=torch.load(r'ckpt_models\wy_runC\checkpoint\035000.pt',weights_only=False); print('gen' in ck, len(ck['gen']))"
copy ckpt_models\wy_runC\checkpoint\035000.pt  wy\checkpoint\new.pt        # 链路A 定版
# 2. 链路B 定版 → audio2lip.pt（E2 拐点）
python -c "import torch; ck=torch.load(r'logs\Audio2Lip\wy\20260916_2123\checkpoint\epoch_001\066526.pt',weights_only=False); print('audio2lip' in ck)"
copy logs\Audio2Lip\wy\20260916_2123\checkpoint\epoch_001\066526.pt  wy\checkpoint\audio2lip.pt  # 链路B 定版（E2）
# 3. 部署后验证（文件名+键+大小三重）：new.pt≈644MB / audio2lip.pt≈11MB
Get-ChildItem wy\checkpoint | Select-Object Name, Length
# 4. 重启 realtime_serve 生效（模型只在启动时加载）；/status 的 available_segments 复核
```
> 工作区漂移防护（Eng A2）：部署源路径以"生成时的日志/记录"为准，勿凭记忆；copy 前的 python 键校验就是防"指向不存在或被覆盖产物"的最后一道闸。大规模重训时建议对定版 ckpt 记 SHA-256（`Get-FileHash`）入训练日志。

### 9.4 模块副本矩阵（编排/改码时勿混用）

| 角色 | 链路A/部署端 | 链路B 训练端 | 关系 |
|---|---|---|---|
| Generator | 根 `networks/generator_lip_pose.py` | `train/networks_Lip_NonLip/generator.py` | state_dict 键实测一致（195 张量），可互换 |
| audio_encoder | 根 `networks/audio_encoder.py`（部署） | `train/networks_audio2lip/audio_encoder.py`（训练，构造自动吃 Wav2Lip.pt） | 同名不同行为 |
| VGG loss | `fine_tune/vgg19.py`（带权重图+no_grad） | `train/vgg19.py`（旧版） | 功能分叉 |
| import networks | 根 networks/ 包 | train/networks/ 包（sys.path 顺序决定） | 两套不同网络包 |

---

## 10. 静默失败登记（高危）

以下失败模式**无异常、无报错**，直接表现为"功能坏了"。整合排查从这张表开始：

| # | 失败 | 触发 | 现象 | 排查路径（可执行命令） |
|---|---|---|---|---|
| 1 | **`/audio/push` 死路径** | 按旧契约推流式分块音频 | 200 正常返回，但无口型；数据堆积内存 | 换 `/audio/push_full`。实测确认：`curl -X POST http://127.0.0.1:8000/audio/push -H "Content-Type: application/json" -d "{\"pcm_base64\":\"AAAA\",\"seq\":0}"` 返回 `{ok,...}` 但口型不动（对照 push_full 同 payload 有口型） |
| 2 | **PCM 规格不符** | 推送非 16kHz/非单声道/非 int16 | 口型错乱或无反应，无报错 | `ffprobe -show_streams <tts输出.wav>` 核对 sample_rate=16000/channels=1/codec=pcm_s16le；不符先 `ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt s16 out.wav` |
| 3 | **渲染线程熔断停摆** | 连续 ~30s 单帧失败 | HTTP 200 正常、画面冻结 | `curl http://127.0.0.1:8000/status` 看 `last_error` 非空即熔断；supervisor 监控该字段并重启进程 |
| 4 | **LMDB 旧特征混入** | 重提特征后未 `prepare_feature_lmdb --force` | 训练"成功"但学的是旧 lip_feature | 比对新鲜度：`ls -l --time-style=full-iso wy\lip_feature\1#1.npy wy\lmdb_features\data.mdb`——lmdb_features 的 mtime 必须晚于全部特征 npy；否则重跑步骤 9 加 `--force` |
| 5 | fastrag 检索空结果 | Config.toml 路径相对目录错拿（CWD 有同名文件） | /query 200 但 answer 兜底话术、context 空 | `curl http://127.0.0.1:11420/health`——`chunks` 数应=知识库实际块数（当前 145）；不符即 Config 错拿，改用 `FASTRAG_CONFIG` 绝对路径重启 |
| 6 | fastrag 批量静默截断 | 用 /batch_query 传多条 | 只有第一条被回答，无任何提示 | 逐条 /query（或临时 `$env:FASTRAG_SERVER__MAX_BATCH_SIZE="8"` 重启验证覆盖生效） |
| 7 | UnityCapture 后端静默回退 | 未注册滤镜但装过 OBS 虚拟摄像头 | 画面出现在 OBS 摄像头而非 UnityCapture 设备 | 查服务启动日志 `_active_backend` 字样；先跑 `UnityCapture-master\Install\Install.bat`（管理员） |
| 8 | 步骤 10 中断续跑行数不一致 | 预处理中断后续跑 | 训练 facemark fail-fast / 服务 ValueError: Data mismatch | `(Get-ChildItem wy\count\1).Count` vs `python -c "import numpy as np; print(len(np.load(r'wy\metadata\1.npy'), axis=None) if np.load(r'wy\metadata\1.npy').ndim>1 else len(np.load(r'wy\metadata\1.npy')))"` 不一致即删净该视频 orifra/count/metadata/facemark 四处重跑 |
| 9 | **push_full 过载无背压**（Eng E1） | 弹幕高峰连续密集推送整句 | 推送 200 但口型越排越晚（延迟累积），无丢弃/熔断 | AI-Vtuber 侧单飞行句 + 队列上限（建议 ≤2）；观察 `/status.buffer_ms` 与 `preroll_state` 持续增长即过载 |

**熔断恢复验证**（整合验收用）：手动 `taskkill /F /PID <realtime_serve pid>` 后 supervisor 应在 ≤30s 内重启并重新就绪（/status preroll_state==complete）。

---

## 11. 与 AI-Vtuber 的对接面汇总

> AI-Vtuber 侧完整改造清单见 `D:\AI\AI-Vtuber\specs\integration\EDTalk实时推理服务对接纪要.md`（A1-A4，T1 spike 产出）。本节给 EDTalk 侧视角的对接面总账。

| 对接项 | AI-Vtuber 侧落点 | EDTalk 侧接口 | 状态 |
|---|---|---|---|
| 服务拉起/停止 | `config.coordination_program[]` 加条目（需先修 frontend/main.py:213 parameters 透传） | 启动命令见 §3.1；`D:\AI\start\start_edtalk.bat` 已存在可参考 | 蓝图就绪 |
| supervisor（拉起/探活/重启/日志重定向） | 新增职责（现无统一管理器） | `/status`（preroll_state==complete 即就绪；last_error 监控）；stdout 重定向 | **缺口**：需实施 |
| 音频数据面 | `playback_manager.py:147/245` 两处 play 旁推 PCM（visual_body=='edtalk_realtime' 条件） | `POST /audio/push_full`，16kHz int16 base64 ≤10MB | 蓝图就绪（A3） |
| 画面消费 | OBS/推流端选 UnityCapture 摄像头 | 虚拟摄像头（§3.8） | 滤镜需 Install.bat 注册一次 |
| 前端控制 tab | `frontend/ui/tabs/visual_body.py` 加 edtalk_realtime 节（与 metahuman_stream 互斥） | /segment/switch、/segment/schedule、/gaze/params、/repair/params、/status；409 时按 `{mode}_capable` 置灰 | 蓝图就绪（A1） |
| 反应段联动 | comment_handler 尾/回复分发侧 | `POST /segment/switch at=segment_end`（连续播报用 set_schedule） | 蓝图就绪（A4） |
| 知识库问答 | comment_handle 责任链新环 + config.fastrag 节 | fastrag HTTP（§4.3）+ LiveStreamBot 模板 | 蓝图就绪（§4.7） |
| 契约回写 | — | EMBEDDING.md 7 处差异（§3.3）→ T3 | 待办 |
| 客户端降级 | 所有调用容忍连接失败（跳过推送不阻断直播主流程） | — | 对接纪要 A4 已规定 |

---

## 12. 整合验收清单

### 12.0 整合前冒烟清单（Eng T2——依赖关系本身的验证，先于一切对接代码）

| # | 冒烟项 | 命令/操作 | 通过判据 |
|---|---|---|---|
| S1 | realtime_serve 裸跑 | `cd D:\AI\EDTalk && runtime312\python.exe -m realtime_serve.main --character-dir wy --segments auto` | 启动无异常；`curl 127.0.0.1:8000/status` 得 preroll_state==complete |
| S2 | push_full 口型 | 用 §10-2 命令验 PCM 后推一句 16kHz wav 的 base64 | 虚拟摄像头可见口型，`/status.buffer_ms` 正常 |
| S3 | fastrag exe 裸跑 | `FASTRAG_CONFIG=<绝对路径>` 下启动 server.exe | `/health` 200 且 chunks=145 |
| S4 | fastrag 查询 | `curl -X POST 127.0.0.1:11420/query -d "{\"question\":\"测试\"}" -H "Content-Type: application/json"` | 200 且 answer 非兜底话术 |
| S5 | 一键训练干跑 | `run_preprocess.py --base_dir wy --dry_run` | 11 步命令全部打印、无脚本缺失报错 |
| S6 | 虚拟摄像头设备 | OBS 源列表出现 UnityCapture 设备 | 有（且 `_active_backend`=unitycapture） |
| S7 | GPT-SoVITS 采样率 | `ffprobe` 任一 TTS 输出 wav | sample_rate=16000（对接纪要已确认，复验防 TTS 配置漂移） |

### 12.1 终态验收（每条含判定方法）

| # | 验收项 | 判定方法 |
|---|---|---|
| 1 | realtime_serve 被 AI-Vtuber 拉起并就绪 | coordination_program 启动后，AI-Vtuber 轮询 `/status` 得 `preroll_state=="complete"` 且 `current_segment` 非空 |
| 2 | TTS 音频→口型闭环 | AI-Vtuber 触发一句 TTS（16kHz wav），`/audio/push_full` 推送后 40ms 级口型同步、虚拟摄像头可见说话 |
| 3 | 画面进直播链 | OBS 选 UnityCapture 摄像头源，画面与声卡音频同步 |
| 4 | 段控制 | 前端 tab 即时切换（≤~0.3s 不连续）与 segment_end 边界无缝切换均生效；/status 回显正确 |
| 5 | gaze/repair 热调 | 启动带 on 时热调生效（next_frame）；启动 off 时开关按 capable 置灰（409） |
| 6 | fastrag 服务托管 | AI-Vtuber 拉起 server.exe（无窗口），/health 探活通过，chunks 数与知识库一致 |
| 7 | 知识库指令闭环 | 弹幕发指定前缀问题 → 责任链命中 → 直播回复为 80 字内知识库答案；fastrag 掉线时降级话术兜底不冷场 |
| 8 | 一键训练 | 单命令完成：预处理（幂等跳过）→ 链路A → eval 门 → 链路B → 部署打包；中断后可续跑 |
| 9 | 训练验收门 | eval_finetune 门槛通过（姿势 MAE ≤11.1px 等）+ mouth_metrics 嘴区指标不劣于 Run C 基线 |
| 10 | 崩溃恢复 | 杀掉 realtime_serve 进程后 supervisor 自动重启并重新就绪 |
| 11 | 文档同步 | 本文变更登记节（§14）随首个 EDTalk 接口变更补记，SHA 更新 |

---

## 13. 附录A：视频通道延迟预算

已裁决先例：虚拟摄像头（UnityCapture DirectShow 共享内存）。延迟构成（4080S 实测锚点，PIPELINE_MAP §9）：

| 环节 | 延迟 | 说明 |
|---|---|---|
| 音频推理（Audio2Lip 分块渐进） | 每块 125 帧推理，块间 4ms 让路 | 推理落后时音频 HOLD 等待（不跳帧不重复） |
| 单帧渲染 gen p50 | 8.9ms | CUDA Graph 捕获后 |
| GPEN 修复 p50 | 7.2ms（adaptive 模式缓冲≥6帧才修） | repair on 时 |
| 输出节奏 | 25fps 固定 40ms tick | OutputStreamer 唯一时钟 |
| 口型领先量 | **≤~40ms（DAC 延迟），绝不滞后** | 帧锁定语义（EMBEDDING v0.2） |
| 虚拟摄像头共享内存 | <1ms/帧 | DirectShow 滤镜本机通信 |
| 消费端（OBS 采集） | 1-2 帧量级 | OBS 合成/编码管线，非本服务控制 |

备选通道对比（备查，不重开决策）：共享内存帧 API（可省 DirectShow 一跳，但需 AI-Vtuber 侧帧消费者改造+崩溃域耦合）；本地 HTTP 视频流（网络栈开销+编解码延迟，最差）。当前实测 25fps 下音画同步达标（2026-09-04 真机验证），虚拟摄像头维持。

---

## 14. 变更登记

> 规则：EDTalk 接口面（EMBEDDING.md 方法面/HTTP 端点、本文档 §3-§9 的参数与路径）发生变更时在此补记一行：日期 + 基线 SHA + 变更点 + 影响小节。

| 日期 | 基线 | 变更 | 影响 |
|---|---|---|---|
| 2026-09-21 | c013414 | 初版（/autoplan 整合准备） | 全文 |
| 2026-09-22 | tag integrate-prep-20260921 (f220a6e) | Eng 阶段修订落地 + T1 基线冻结 | §2.2/§3.3/§4.5/§8.2/§9.1/§9.3/§10/§12.0/文档头 |

---

## 15. 附录B：引用文档索引

| 文档 | 内容 | 关系 |
|---|---|---|
| `realtime_serve/EMBEDDING.md` | 嵌入契约 v0.3（方法面/HTTP/语义/拓扑） | **单一来源**，本文引用不重写 |
| `realtime_serve/PIPELINE_MAP.md` | 管线全图（线程/锁/时序） | 实现细节出处 |
| `realtime_serve/RUNBOOK.md` | 运行手册（eye_source/启动/排障） | 运维细节出处 |
| `D:\AI\AI-Vtuber\specs\integration\EDTalk实时推理服务对接纪要.md` | A1-A4 改造清单 | AI-Vtuber 侧实施依据 |
| `D:\AI\AI-Vtuber\specs\integration\统一环境依赖整合计划.md` | 统一环境+submodule 策略（含已裁决偏差） | 环境与收编路线出处 |
| `DOCS/01~03_微调训练流程系列.md` | fine_tune/解耦预训练机制 | 训练机制细节 |
| `DOCS/单人微调训练完整指南.md`、`DOCS/推理与微调脚本使用指南.md` | 操作手册 | 本文的参数补充 |
| `FINE_TUNE_FIX_PLAN.md` | 微调修复计划 v3 | 配方依据 |
| `fastrag/README.md`、`fastrag/知识库规范.md` | fastrag 官方文档（⚠ 环境变量写法有误，以本文 §4.2 实测为准） | fastrag 操作细节 |
| 项目记忆档案（edtalk-audio2mouth-optimize-final 等） | 配方诊断链与消融结论 | §9.2 配方出处 |
