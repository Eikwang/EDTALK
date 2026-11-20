"""
query.py - 交互式查询脚本
用于测试知识库问答功能
"""
# @ACTIVE — fastrag 项目交互式查询入口

import os
import sys
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import INDEX_DIR, HISTORY_MAX_MESSAGES
from fastrag_native import fastragNative, pick_best_reference


async def main():
    print("=" * 60)
    print("fastrag 交互式查询")
    print("=" * 60)

    # 检查索引是否存在
    if not os.path.exists(INDEX_DIR):
        print(f"\n错误: 索引目录不存在: {INDEX_DIR}")
        print("请先运行 build_index_native.py 构建索引。")
        return

    # 初始化 RAG（加载已有索引）
    print("\n加载模型和索引...")
    rag = fastragNative()
    rag.load(str(INDEX_DIR))
    print(f"已加载 {len(rag.chunks)} 个文本块")

    # 交互式查询
    print("\n" + "=" * 60)
    print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    # 多轮对话历史（指代消解 + 上下文）
    history = []

    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break

        # 执行查询（带历史）
        result = await rag.query(question, history=history)

        print("\n" + "-" * 60)
        print(f"[路由] {result.get('route_mode', '?')}")
        print(f"答案: {result['answer']}")

        # 维护历史供下一轮使用（assistant 只携带该轮 rerank 分数最高的一条参考，
        # 追问时注入 prompt 做"上下文携带"，残句追问才能对准上一轮产品）
        history.append({"role": "user", "content": question})
        history.append({
            "role": "assistant",
            "content": result["answer"],
            "references": pick_best_reference(result.get("context", [])),
        })
        history = history[-HISTORY_MAX_MESSAGES:]
        # print("\n相关文档片段:")
        # for i, src in enumerate(result["context"], 1):
        #     print(f"\n[{i}] (score: {src['score']:.3f})")
        #     content = src['content'][:100]
        #     print(f"   {content}...")


if __name__ == "__main__":
    asyncio.run(main())