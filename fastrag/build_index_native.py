"""
build_index_native.py - 索引构建脚本
扫描知识库 Markdown 文档，面包屑分块 + FAISS 向量索引
"""
# @ACTIVE — fastrag 项目索引构建入口

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import KB_DIR, INDEX_DIR
from fastrag_native import fastragNative


def load_knowledge_base(kb_dir: str):
    """加载知识库目录下的所有文档"""
    documents = []

    if not os.path.exists(kb_dir):
        print(f"知识库目录不存在: {kb_dir}")
        return documents

    for root, dirs, files in os.walk(kb_dir):
        for file in files:
            if file.endswith((".md", ".txt")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            documents.append({
                                "path": file_path,
                                "content": content
                            })
                            print(f"  [OK] 已加载: {file_path}")
                except Exception as e:
                    print(f"  [FAIL] 读取失败 {file_path}: {e}")

    return documents


def main():
    print("=" * 60)
    print("知识库索引构建（面包屑分块 + FAISS 向量索引）")
    print("=" * 60)

    if not os.path.exists(KB_DIR):
        print(f"\n创建知识库目录: {KB_DIR}")
        os.makedirs(KB_DIR, exist_ok=True)
        print("\n请将您的知识库文档添加到上述目录后重新运行此脚本。")
        return

    print(f"\n扫描知识库目录: {KB_DIR}")
    documents = load_knowledge_base(str(KB_DIR))

    if not documents:
        print("\n知识库为空，请添加文档后重试。")
        return

    print(f"\n共加载 {len(documents)} 个文档")

    print("\n初始化 fastrag 推理引擎...")
    rag = fastragNative()

    print("\n构建索引...")
    total_chunks = 0
    for doc in documents:
        before = len(rag.chunks)
        rag.insert_document(doc["content"], source=doc["path"])
        added = len(rag.chunks) - before
        total_chunks += added
        doc_name = os.path.basename(doc["path"])
        print(f"  [{doc_name}]: {added} 块")

    print("\n保存索引...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    rag.save(str(INDEX_DIR))

    print("\n" + "=" * 60)
    print("索引构建完成！")
    print("=" * 60)
    print(f"\n索引位置: {INDEX_DIR}")
    print(f"文档数量: {len(documents)}")
    print(f"文本块数: {len(rag.chunks)}（平均每文档 {len(rag.chunks)/len(documents):.1f} 块）")


if __name__ == "__main__":
    main()
