"""
server.py - fastrag FastAPI 服务
模型常驻内存，支持多客户端并发查询
"""
# @ACTIVE — fastrag 项目 API 服务器

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from config import INDEX_DIR, SERVER_PORT
from fastrag_native import fastragNative, pick_best_reference

# ===== Pydantic 模型 =====

class HistoryMessage(BaseModel):
    role: str = "user"       # "user" 或 "assistant"
    content: str
    references: Optional[List[str]] = None  # assistant 条目可带该轮命中参考（只传最高分一条，上下文携带）

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[HistoryMessage]] = None  # 多轮对话历史（指代消解 + 上下文）

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: List[Dict]
    route_mode: str = "kb"  # 路由结果："kb"=知识库问答，"chat"=自由闲聊
    references: List[str] = []  # 本轮 rerank 最高分的一条参考（客户端应随下一轮 history 回传）

class HealthResponse(BaseModel):
    status: str
    chunks: int
    rerank_enabled: bool
    llm_quant: str
    reranker_quant: str
    vram_gb: float

# ===== 全局变量 =====

rag: Optional[fastragNative] = None

# ===== Lifespan 事件 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期：启动时加载模型和索引，关闭时清理"""
    global rag

    print("\n" + "=" * 60)
    print("fastrag 服务启动中...")
    print("=" * 60)

    try:
        # 检查索引是否存在
        if not os.path.exists(INDEX_DIR):
            print(f"\n错误: 索引目录不存在: {INDEX_DIR}")
            print("请先运行 build_index_native.py 构建索引。")
            raise RuntimeError(f"索引目录不存在: {INDEX_DIR}")

        # 初始化 RAG（加载模型）
        print("\n[1/2] 加载模型...")
        rag = fastragNative()
        print("      模型加载完成")

        # 加载索引
        print("\n[2/2] 加载索引...")
        rag.load(str(INDEX_DIR))

        print(f"\n[OK] 服务启动完成！")
        print(f"   - 文本块: {len(rag.chunks)}")
        print(f"   - Rerank: {'启用' if rag.enable_rerank else '禁用'}")
        print(f"   - 端口: {SERVER_PORT}")
        print("=" * 60 + "\n")

        yield  # 服务运行中

    except Exception as e:
        print(f"\n[FAIL] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        print("\n服务关闭中...")
        if rag and rag.use_gpu:
            import torch
            torch.cuda.empty_cache()
        rag = None
        print("已清理资源")

# ===== 创建 FastAPI 应用 =====

app = FastAPI(title="fastrag Service", version="2.0.0", lifespan=lifespan)

# ===== API 路由 =====

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查 — 返回服务状态、索引信息和显存占用"""
    if rag is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    import torch
    from config import LLM_QUANT_BITS, RERANKER_QUANT_BITS

    vram = torch.cuda.memory_allocated(0) / 1024**3 if rag.use_gpu else 0.0

    return HealthResponse(
        status="ok",
        chunks=len(rag.chunks),
        rerank_enabled=rag.enable_rerank and rag.reranker_model is not None,
        llm_quant=f"INT{LLM_QUANT_BITS}" if LLM_QUANT_BITS < 16 else "BF16",
        reranker_quant=f"INT{RERANKER_QUANT_BITS}" if RERANKER_QUANT_BITS < 16 else "BF16",
        vram_gb=round(vram, 2)
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口 — 输入问题，返回答案和相关上下文

    请求体:
        question: 客户问题（必填）

    返回:
        question: 原始问题
        answer: AI 客服回答
        context: 相关文档片段列表
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        history = [
            {"role": m.role, "content": m.content, **({"references": m.references} if m.references else {})}
            for m in (request.history or [])
        ]
        result = await rag.query(request.question, history=history)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            context=result["context"],
            route_mode=result.get("route_mode", "kb"),
            references=pick_best_reference(result.get("context", [])),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_query")
async def batch_query(questions: List[str]):
    """批量查询接口 — 输入问题列表，返回答案列表

    请求体:
        ["问题1", "问题2", ...]

    返回:
        {"results": [{"question": "...", "answer": "..."}, ...]}
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    MAX_BATCH_SIZE = 5
    if len(questions) > MAX_BATCH_SIZE:
        questions = questions[:MAX_BATCH_SIZE]

    results = []
    for q in questions:
        result = await rag.query(q)
        results.append({
            "question": q,
            "answer": result["answer"]
        })
    return {"results": results}

@app.post("/clear_cache")
async def clear_cache():
    """清理 GPU 缓存"""
    if rag is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    if rag.use_gpu:
        import torch
        torch.cuda.empty_cache()
        vram = torch.cuda.memory_allocated(0) / 1024**3
        return {"status": "ok", "message": "缓存已清理", "vram_gb": round(vram, 2)}
    else:
        return {"status": "ok", "message": "CPU 模式无需清理"}

# ===== 主入口 =====

if __name__ == "__main__":
    print(f"\nfastrag 服务将启动在 http://localhost:{SERVER_PORT}")
    print(f"API 文档: http://localhost:{SERVER_PORT}/docs")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info",
        workers=1,
        limit_concurrency=50,
    )
