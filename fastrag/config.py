"""
config.py - 配置文件
管理模型路径、检索参数、角色提示词、LLM 参数等
"""
# @ACTIVE — fastrag 项目配置文件

from pathlib import Path

# ===== 路径配置 =====
BASE_DIR = Path(__file__).parent
# 模型在父目录的 chekpoint 中
MODEL_DIR = BASE_DIR / "chekpoint"

# 模型路径（chekpoint 下可测试的组合：
#   LLM:      MiniCPM5-1B | Qwen2.5-0.5B-Instruct
#   Embedder: ritrieve_zh_v1 | Qwen3-Embedding-0.6B
#   Reranker: mxbai-rerank-base-v2 | Qwen3-Reranker-0.6B
# 系统按模型目录结构自动识别协议，任意组合可直接运行）
LLM_MODEL_PATH = MODEL_DIR / "Qwen2.5-0.5B-Instruct"  # 主 LLM
EMBEDDER_MODEL_PATH = MODEL_DIR / "ritrieve_zh_v1"  # Embedding 模型
RERANKER_MODEL_PATH = MODEL_DIR / "mxbai-rerank-base-v2"  # Reranker 模型
# Qwen3-Embedding 的查询指令（仅 lasttoken 型 embedder 使用；文档侧不加指令）
EMBED_QUERY_INSTRUCTION = (
    "这是正在直播带货中的客户咨询，"
    "请检索商品资料中能回答该问题或与问题相关的段落信息"
)
# 调试开关：打印发送给 LLM 的完整消息（system/user 全文 + 生成参数）
# 交互测试设 True 便于观察 prompt；生产环境建议 False
DEBUG_LLM_MESSAGES = True

# ===== 检索参数 =====
TOP_K_CANDIDATES = 4   # 向量检索候选数量（多召回一些，供追问时二次 rerank 择优）
MAX_CHUNK_CHARS = 128  # 单块正文最大字符数（超过则二次切分，避免 embedding 截断丢失尾部信息）
ENABLE_RERANK = True    # 是否启用 rerank 重排序
RERANK_TOP_K = 2        # rerank 后送入 LLM 的 top-k

# ===== 相关性门控（路由）=====
# 单阈值路由：rerank top1 分数 ≥ 此值 → 知识库模式（注入参考资料+贪心解码）
#            分数 <  此值 → 闲聊模式（不注入资料+采样解码）
# mxbai 实测：真实 KB 问题 ≥0.99；闲聊集中在 0.5~0.72（偶有"你是谁"0.95 类虚高）。
# 阈值需用真实数据的分数分布标定（正例/负例断层处）；切换 reranker 组合后必须重新标定
RERANK_THRESHOLD = 0.9

# ===== 生成失败兜底 =====
# 1B 模型偶发空白/复读答复，重试后仍无效时使用
KB_FALLBACK_ANSWER = "这个资料里没有提到，我不太清楚呢。"
CHAT_FALLBACK_ANSWER = "哎呀，这个我一时说不上来，咱们换个话题聊聊吧！"

# ===== 多轮对话 =====
HISTORY_MAX_MESSAGES = 2    # 注入生成 prompt 的历史消息上限（user+assistant 合计）
RETRIEVAL_HISTORY_TURNS = 1  # 拼接进检索 query 的最近用户发言轮数（指代消解：
                             # "是啥牌子的"单独检索会命中错误产品，需拼接上文）

# ===== 量化配置（显存优化）=====
# Embedder 运行设备："cpu"=CPU推理, "cuda-fp16"=GPU FP16推理, "cuda"=GPU FP32推理
# CPU: 显存最低(~1.8GB)，但CPU被其他任务占用时会变慢
# cuda-fp16: 显存适中(~2.4GB)，GPU加速8-20倍，推荐生产环境使用
# cuda: 显存较高(~3.2GB)，不推荐
EMBEDDER_DEVICE = "cuda-fp16"
# LLM 量化：4=INT4（省约 1.0GB），8=INT8（省约 0.5GB），16=BF16 无量化
LLM_QUANT_BITS = 16
# Reranker 量化：4=INT4（省约 0.3GB），8=INT8（省约 0.3GB），16=BF16 无量化
RERANKER_QUANT_BITS = 16

# ===== 知识库配置 =====
KB_DIR = BASE_DIR / "knowledgebase"  # 知识库文档目录
INDEX_DIR = BASE_DIR / "kbindex"      # 索引存储目录

# ===== 角色提示词 =====
# 系统角色设定，用于定义 AI 客服的人设和行为规范
SYSTEM_ROLE = "你叫老王，你是一位专业、友好的线上销售主播，现在正在直播间与观众互动。"

# ===== 统一回答规则 =====
# KB 与闲聊共用一套提示词，第 5/6 条按"是否提供参考资料"自适应两种模式。
# 注意：0.5B 模型对规则遵从性很强，精简规则（尤其删掉接地条款）会大幅降低回答质量，
# 规则保持完整，不要随意删减。
SYSTEM_RULES = """
1.你必须完全沉浸在你的人设中！像真人一样用流畅生动的语气回答问题。
2.使用中文自然语句进行口语化回答、不能包含任何特殊字符或类似表格、列表等书面格式。
3.严禁使用任何AI或机器人的套话！不要复述用户的问题，不要输出开场白和问候语。
4.直接给出答案；内容要简洁，尽量不超过{max_answer_length}字。
5.如果提供了`参考资料`,必须完全基于参考资料的内容进行回答，`参考资料`中的信息永远优先于你自己的知识.
6,在存在`参考资料`时，不要在回答中包含'根据参考资料'、'资料显示'等声明。
7.如果没有提供`参考资料`,可以根据`对话历史`中的`来源`信息，并结合你自己的知识自然回答。
8.当问题涉及时间、日期，直接使用`时间信息`里的信息回答。
"""

# ===== LLM 生成参数（知识库模式）=====
MAX_ANSWER_LENGTH = 80      # 答案最大字符数
LLM_MAX_NEW_TOKENS = 256  # LLM 最大生成 token 数
LLM_TEMPERATURE = 0.5       # 生成温度（越低越确定，越高越随机）
LLM_TOP_P = 0.9             # Top-p 采样阈值
LLM_DO_SAMPLE = False        # 是否启用采样（False 为贪心解码，知识库问答保持贪心）
LLM_REPETITION_PENALTY = 1.1  # 防复读惩罚（抑制 1B 模型偶发的复读/回声退化）

# ===== 闲聊模式生成参数 =====
# 与知识库无关的闲聊需要一定随机性，回复才自然，不重复死板
CHAT_MAX_ANSWER_LENGTH = 120      # 闲聊答案最大字符数
CHAT_LLM_MAX_NEW_TOKENS = 256     # 闲聊最大生成 token 数
CHAT_LLM_TEMPERATURE = 0.7        # 采样温度
CHAT_LLM_TOP_P = 0.9              # Top-p 采样阈值
CHAT_LLM_DO_SAMPLE = True         # 闲聊启用采样

# ===== 服务配置 =====
SERVER_PORT = 11420         # API 服务端口
