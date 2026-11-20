"""
api_client.py - fastrag API 对接示例
用于直播系统接入 fastrag 知识库问答服务

重要：fastrag 服务是无状态的！多轮对话历史由客户端维护，
每次请求把 history 回传，assistant 条目需携带上一轮的 references
（上下文携带，残句追问"是啥牌子的"才能对准上一轮产品）。

使用方式:
    from api_client import LiveStreamBot
    bot = LiveStreamBot()
    if bot.is_ready():
        print(bot.answer("新达抽纸一件是几包"))      # 第 1 轮
        print(bot.answer("是啥牌子的"))              # 第 2 轮（自动带历史）
"""
import requests
import time


# ===== 配置 =====
FASTRAG_URL = "http://localhost:11420"


def _trim_rounds(history: list, max_rounds: int) -> list:
    """客户端侧按完整轮次裁剪历史（问题+答案成对保留最近 max_rounds 轮）"""
    if not history or max_rounds <= 0:
        return []
    keep = history[-(max_rounds * 2):]
    while keep and keep[0].get("role") != "user":  # 丢弃开头孤立的 assistant
        keep.pop(0)
    return keep


def query(question: str, history: list | None = None, timeout: int = 60) -> str:
    """单条查询

    Args:
        question: 客户问题
        history: 对话历史（可选，格式 [{"role","content","references"?}]；
                 多轮对话时必须回传，否则追问无法指代消解）
        timeout: 超时秒数（默认 60s，LLM 推理约需 2-3s）

    Returns:
        AI 回答文本
    """
    payload = {"question": question}
    if history:
        payload["history"] = history
    resp = requests.post(f"{FASTRAG_URL}/query", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["answer"]


def query_full(question: str, history: list | None = None, timeout: int = 60) -> dict:
    """查询并返回完整结果（含 answer/context/route_mode/references）

    Returns:
        {"question": "...", "answer": "...", "route_mode": "kb|chat",
         "references": ["本轮最高分参考"], "context": [...]}
    """
    payload = {"question": question}
    if history:
        payload["history"] = history
    resp = requests.post(f"{FASTRAG_URL}/query", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def batch_query(questions: list[str], timeout: int = 120) -> list[str]:
    """批量查询

    Returns:
        答案列表
    """
    resp = requests.post(
        f"{FASTRAG_URL}/batch_query",
        json=questions,
        timeout=timeout,
    )
    resp.raise_for_status()
    return [r["answer"] for r in resp.json()["results"]]


def health_check(timeout: int = 5) -> dict:
    """健康检查

    Returns:
        {"status": "ok", "chunks": 145, "llm_dtype": "BF16", ...}
    """
    resp = requests.get(f"{FASTRAG_URL}/health", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def clear_cache(timeout: int = 5) -> dict:
    """清理模型 KV cache，释放显存"""
    resp = requests.post(f"{FASTRAG_URL}/clear_cache", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ===== 直播系统集成 =====

class LiveStreamBot:
    """直播 AI 客服集成（内置多轮历史维护，用法与 query.py 交互模式一致）

        bot = LiveStreamBot()
        if bot.is_ready():
            print(bot.answer("新达抽纸一件是几包"))   # 第 1 轮
            print(bot.answer("是啥牌子的"))           # 追问自动带历史+上下文携带
            bot.reset()                              # 换话题/新观众时清空历史
    """

    def __init__(self, server_url: str = FASTRAG_URL, max_history_rounds: int = 1):
        self.server_url = server_url
        self.max_history_rounds = max_history_rounds  # 保留最近几轮完整对话
        self.history: list = []  # [{"role","content","references"?}]

    def reset(self) -> None:
        """清空对话历史（换话题/新观众会话时调用）"""
        self.history = []

    def is_ready(self) -> bool:
        """检查服务是否就绪"""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=3)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except requests.RequestException:
            return False

    def answer(self, question: str) -> str:
        """回答客户问题（核心方法），内部自动维护多轮历史"""
        try:
            payload = {"question": question}
            if self.history:
                payload["history"] = self.history
            resp = requests.post(
                f"{self.server_url}/query",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            return "抱歉，系统响应超时，请稍后再问哦~"
        except requests.ConnectionError:
            return "抱歉，客服系统暂时离线，请稍后再试~"
        except Exception:
            return "抱歉，出了点小问题，请稍后再问哦~"

        # 请求成功才记入历史；assistant 携带该轮最高分参考（上下文携带）
        self.history.append({"role": "user", "content": question})
        entry = {"role": "assistant", "content": data.get("answer", "")}
        refs = [r for r in (data.get("references") or []) if r]
        if refs:
            entry["references"] = refs[:1]
        self.history.append(entry)
        self.history = _trim_rounds(self.history, self.max_history_rounds)
        return data.get("answer", "")


# ===== 自测入口 =====

if __name__ == "__main__":
    print("=" * 60)
    print("fastrag API 对接测试")
    print("=" * 60)

    # 1. 健康检查
    print("\n[1] 健康检查...")
    try:
        info = health_check()
        print(f"    状态:   {info['status']}")
        print(f"    文本块: {info['chunks']}")
        print(f"    Rerank: {'启用' if info['rerank_enabled'] else '禁用'}")
        print(f"    LLM:    {info['llm_quant']}")
        print(f"    Reranker: {info['reranker_quant']}")
    except requests.ConnectionError:
        print("    [FAIL] 无法连接服务，请先启动: cd fastrag && server.exe")
        exit(1)
    except Exception as e:
        print(f"    [FAIL] {e}")
        exit(1)

    # 2. 单条查询
    print("\n[2] 单条查询测试...")
    questions = [
        "新达抽纸一件是几包",
        "B级擦手纸一包多少抽",
        "去污液保质期多久",
        "发货到佛山要多久",
    ]

    for q in questions:
        t0 = time.time()
        try:
            answer = query(q)
            elapsed = time.time() - t0
            print(f"    Q: {q}")
            print(f"    A: {answer}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    Q: {q} → [FAIL] {e}")

    # 3. 批量查询
    print("\n[3] 批量查询测试...")
    batch_qs = ["粘鼠板什么牌子", "洗洁精有哪些"]
    t0 = time.time()
    try:
        answers = batch_query(batch_qs)
        elapsed = time.time() - t0
        for q, a in zip(batch_qs, answers):
            print(f"    Q: {q} → A: {a}")
        print(f"    耗时: {elapsed:.1f}s")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # 4. LiveStreamBot 集成示例（含多轮追问演示）
    print("\n[4] LiveStreamBot 多轮对话测试...")
    bot = LiveStreamBot()
    if bot.is_ready():
        for q in ["新达抽纸一件是几包", "是啥牌子的"]:
            answer = bot.answer(q)
            print(f"    Q: {q}")
            print(f"    A: {answer}")
    else:
        print("    服务未就绪")

    # 5. 清理缓存
    print("\n[5] 清理 KV Cache...")
    try:
        result = clear_cache()
        print(f"    {result['message']}")
    except Exception as e:
        print(f"    [FAIL] {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
