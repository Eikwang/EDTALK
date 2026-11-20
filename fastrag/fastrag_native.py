"""
fastrag_native.py - fastrag 推理引擎
面包屑分块 + FAISS 向量检索 + mxbai-rerank 重排序 + LLM 生成
"""
# @ACTIVE — fastrag 项目核心推理引擎

import os
import sys
import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import faiss

from config import (
    LLM_MODEL_PATH, EMBEDDER_MODEL_PATH, RERANKER_MODEL_PATH, INDEX_DIR,
    TOP_K_CANDIDATES, MAX_CHUNK_CHARS, ENABLE_RERANK, RERANK_TOP_K, MAX_ANSWER_LENGTH,
    EMBEDDER_DEVICE, LLM_QUANT_BITS, RERANKER_QUANT_BITS,
    SYSTEM_ROLE, SYSTEM_RULES,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_TOP_P, LLM_DO_SAMPLE,
    RERANK_THRESHOLD,
    KB_FALLBACK_ANSWER, CHAT_FALLBACK_ANSWER, LLM_REPETITION_PENALTY,
    CHAT_MAX_ANSWER_LENGTH, CHAT_LLM_MAX_NEW_TOKENS,
    CHAT_LLM_TEMPERATURE, CHAT_LLM_TOP_P, CHAT_LLM_DO_SAMPLE,
    HISTORY_MAX_MESSAGES, RETRIEVAL_HISTORY_TURNS,
    EMBED_QUERY_INSTRUCTION, DEBUG_LLM_MESSAGES,
)


def pick_best_reference(context: List[Dict]) -> List[str]:
    """从候选中挑 rerank 分数最高的一条作为"来源"（上下文携带只带最优参考，
    多条参考会稀释 0.5B 模型的注意力且增加 prompt 长度）
    """
    if not context:
        return []
    best = max(context, key=lambda c: c.get("rerank_score", c.get("score", 0.0)))
    content = (best.get("content") or "").strip()
    return [content] if content else []


def chunking_by_markdown_headers(content: str, source: str = "") -> List[Dict]:
    """按 Markdown 标题分块，每个块包含完整的层级路径（面包屑）

    分块规则：
    1. # H1 为文档/产品名
    2. ## H2 为主分块边界（分类维度：如"产品参数"、"使用说明"）
    3. ### H3 为二级分块边界（细化参数：如"体积"、"重量"）
    4. 每个块首行包含完整面包屑：# H1 > H2 > H3，嵌入向量自带全路径

    示例输出：
        # 去污液 > 基本参数 > 体积
        长：165mm
        高：368mm
        容量：1.8L

    Args:
        content: Markdown 文档内容
        source: 文档来源路径

    Returns:
        分块列表，每项包含 content, source, doc_title, breadcrumb
    """
    results = []

    # ── 解析 H1 ──
    h1_title = ""
    h1_match = re.match(r'^#\s+(.+)', content)
    if h1_match:
        h1_title = h1_match.group(1).strip()

    # ── 解析各 H2 段落 ──
    h2_sections = re.split(r'\n(?=##\s+)', content)

    for h2_section in h2_sections:
        h2_section = h2_section.strip()
        if not h2_section:
            continue

        # 提取 H2 标题
        h2_title = ""
        h2_match = re.match(r'^##\s+(.+)', h2_section)
        if h2_match:
            h2_title = h2_match.group(1).strip()
        else:
            # 无 H2 标题（H1 和第一个 H2 之间的引言段落）
            lines = h2_section.split('\n')
            non_title_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            if not non_title_lines:
                continue
            intro_text = '\n'.join(non_title_lines).strip()
            if len(intro_text) < 30:
                continue
            h2_title = "概述"
            h2_body = h2_section

        # 提取 H2 正文（去掉 ## 标题行）
        if h2_match:
            h2_body_lines = h2_section.split('\n')
            h2_body = '\n'.join(h2_body_lines[1:]).strip()
        else:
            h2_body = h2_section

        if not h2_body:
            continue

        # ── 尝试按 H3 拆分 ──
        h3_sections = re.split(r'\n(?=###\s+)', h2_body)
        has_h3 = len(h3_sections) > 1

        if has_h3:
            for h3_part in h3_sections:
                h3_part = h3_part.strip()
                if not h3_part:
                    continue

                h3_title = ""
                h3_match = re.match(r'^###\s+(.+)', h3_part)
                if h3_match:
                    h3_title = h3_match.group(1).strip()
                    h3_body = '\n'.join(h3_part.split('\n')[1:]).strip()
                else:
                    h3_title = ""
                    h3_body = h3_part

                if not h3_body:
                    continue

                parts = [p for p in [h1_title, h2_title, h3_title] if p]
                breadcrumb = " > ".join(parts)

                results.append({
                    "content": f"# {breadcrumb}\n{h3_body}",
                    "source": source,
                    "doc_title": h1_title,
                    "breadcrumb": breadcrumb,
                    "chunk_order_index": len(results),
                })
        else:
            parts = [p for p in [h1_title, h2_title] if p]
            breadcrumb = " > ".join(parts)

            results.append({
                "content": f"# {breadcrumb}\n{h2_body}",
                "source": source,
                "doc_title": h1_title,
                "breadcrumb": breadcrumb,
                "chunk_order_index": len(results),
            })

    # ── 处理无结构文档（无 H2：仅有 H1 或纯段落文本）──
    if not results:
        body = content
        h1_line_end = content.find('\n')
        if h1_line_end > 0 and content[:h1_line_end].strip().startswith('#'):
            body = content[h1_line_end:].strip()
        if body:
            breadcrumb = h1_title or "未命名"
            results.append({
                "content": f"# {breadcrumb}\n{body}",
                "source": source,
                "doc_title": h1_title,
                "breadcrumb": breadcrumb,
                "chunk_order_index": 0,
            })

    # ── 超长块二次切分（避免 embedding 512 token 截断丢失尾部信息）──
    results = _split_long_chunks(results, MAX_CHUNK_CHARS)

    return results


def _split_long_chunks(chunks: List[Dict], max_len: int) -> List[Dict]:
    """把正文超过 max_len 的块按行边界二次切分，每段继承面包屑头

    切分策略：按行聚合，接近 max_len 就封段；多段时在面包屑头追加（第x/y段）
    """
    out = []
    for c in chunks:
        lines = c["content"].split("\n")
        header, body = lines[0], "\n".join(lines[1:])
        if len(body) <= max_len:
            out.append(c)
            continue

        segments, cur, cur_len = [], [], 0
        for line in body.split("\n"):
            if cur and cur_len + len(line) + 1 > max_len:
                segments.append("\n".join(cur))
                cur, cur_len = [], 0
            cur.append(line)
            cur_len += len(line) + 1
        if cur:
            segments.append("\n".join(cur))

        for j, seg in enumerate(segments):
            suffix = f"（第{j + 1}/{len(segments)}段）" if len(segments) > 1 else ""
            out.append({
                "content": f"{header}{suffix}\n{seg}",
                "source": c.get("source", ""),
                "doc_title": c.get("doc_title", ""),
                "breadcrumb": c.get("breadcrumb", ""),
                "chunk_order_index": 0,
            })

    for i, c in enumerate(out):
        c["chunk_order_index"] = i
    return out


class fastragNative:
    """fastrag 原生架构：面包屑分块 + 纯向量检索 + Rerank + LLM 生成"""

    def __init__(self, use_gpu=True, enable_rerank: bool = None):
        print("=" * 60)
        print("初始化 fastrag 原生架构")
        print("=" * 60)

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        self._inference_lock = asyncio.Lock()

        self.enable_rerank = ENABLE_RERANK if enable_rerank is None else enable_rerank

        # 模型
        self.llm = None
        self.llm_tokenizer = None
        self.embedder_model = None
        self.embedder_tokenizer = None
        self.reranker_model = None
        self.reranker_tokenizer = None

        # 知识库存储
        self.chunks: List[Dict] = []
        self.chunk_index = None  # FAISS 文本块向量索引
        self._chunk_hashes = set()  # 内容哈希（跨文档去重）

        # 加载模型
        self._load_models()
        print("\n[OK] 所有模型加载完成！\n")

    def _load_models(self):
        """加载模型（量化 + CPU 分配，优化显存）"""
        from transformers import BitsAndBytesConfig

        # ── 1. Embedder ──
        # EMBEDDER_DEVICE: "cpu" / "cuda-fp16" / "cuda"
        emb_cfg = EMBEDDER_DEVICE.lower()
        if emb_cfg == "cuda-fp16" and self.use_gpu:
            emb_device = "cuda"
            emb_dtype = torch.float16
            emb_label = "GPU FP16"
        elif emb_cfg == "cuda" and self.use_gpu:
            emb_device = "cuda"
            emb_dtype = torch.float32
            emb_label = "GPU FP32"
        else:
            emb_device = "cpu"
            emb_dtype = torch.float32
            emb_label = "CPU"
        print(f"[1/3] 加载 Embedder ({EMBEDDER_MODEL_PATH.name}) → {emb_label}...")
        self.embedder_tokenizer = AutoTokenizer.from_pretrained(
            str(EMBEDDER_MODEL_PATH), trust_remote_code=True
        )
        self.embedder_model = AutoModel.from_pretrained(
            str(EMBEDDER_MODEL_PATH), trust_remote_code=True
        )
        if emb_dtype == torch.float16:
            self.embedder_model = self.embedder_model.half()
        self.embedder_model = self.embedder_model.to(emb_device)
        self.embedder_model.eval()
        self._embedder_device = emb_device

        # 按模型目录结构自动识别协议（支持 ritrieve_zh_v1 / Qwen3-Embedding 等两类）
        # - lasttoken: Qwen3-Embedding（last-token 池化，查询侧带 Instruct 指令）
        # - dense:     sentence-transformers 结构（mean 池化 + 2_Dense 线性层，如 ritrieve_zh_v1）
        # - 其他:      CLS 池化兜底
        self._emb_lasttoken = False
        self._emb_dense = None
        pool_cfg_path = EMBEDDER_MODEL_PATH / "1_Pooling" / "config.json"
        if pool_cfg_path.exists():
            pool_cfg = json.loads(pool_cfg_path.read_text(encoding="utf-8"))
            self._emb_lasttoken = bool(pool_cfg.get("pooling_mode_lasttoken"))
        dense_dir = EMBEDDER_MODEL_PATH / "2_Dense"
        if dense_dir.exists() and not self._emb_lasttoken:
            dcfg = json.loads((dense_dir / "config.json").read_text(encoding="utf-8"))
            from safetensors.torch import load_file
            sd = load_file(str(dense_dir / "model.safetensors"))
            linear = torch.nn.Linear(
                dcfg["in_features"], dcfg["out_features"], bias=dcfg.get("bias", True)
            )
            linear.load_state_dict(
                {k.replace("linear.", ""): v for k, v in sd.items()}, strict=True
            )
            self._emb_dense = linear.to(device=emb_device, dtype=emb_dtype)
            self._emb_dense.eval()

        if self._emb_lasttoken:
            emb_mode, emb_dim = "last-token+查询指令", self.embedder_model.config.hidden_size
        elif self._emb_dense is not None:
            emb_mode, emb_dim = "mean+Dense", self._emb_dense.out_features
        else:
            emb_mode, emb_dim = "CLS", self.embedder_model.config.hidden_size
        self._emb_dim = emb_dim
        emb_vram = self.embedder_model.get_memory_footprint() / 1024**3
        print(f"      -> 池化: {emb_mode}, 维度: {emb_dim}, 设备: {emb_label}, VRAM: ~{emb_vram:.2f} GB")

        # ── 2. LLM ──
        quant_label = f"INT{LLM_QUANT_BITS}" if LLM_QUANT_BITS < 16 else "BF16"
        print(f"[2/3] 加载 LLM ({LLM_MODEL_PATH.name}) — {quant_label}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            str(LLM_MODEL_PATH), trust_remote_code=True, local_files_only=True
        )
        if LLM_QUANT_BITS == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                str(LLM_MODEL_PATH),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
        elif LLM_QUANT_BITS == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                str(LLM_MODEL_PATH),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                str(LLM_MODEL_PATH),
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
        vram = self.llm.get_memory_footprint() / 1024**3
        print(f"      -> VRAM: ~{vram:.2f} GB, 量化: {quant_label}")

        # ── 3. Reranker ──
        # 两种协议自动识别：
        # - qwen3 架构（Qwen3-Reranker）: <Instruct>/<Query>/<Document> 格式，
        #   取末位 yes/no logit 差做 sigmoid 得 0~1 分数
        # - 其他（mxbai-rerank）: "query/document + Relevance:" 格式，
        #   取末位 "0"/"1" token 的 softmax 概率
        if self.enable_rerank and RERANKER_MODEL_PATH.exists():
            r_quant_label = f"INT{RERANKER_QUANT_BITS}" if RERANKER_QUANT_BITS < 16 else "BF16"
            arch = ""
            r_cfg_path = RERANKER_MODEL_PATH / "config.json"
            if r_cfg_path.exists():
                arch = (json.loads(r_cfg_path.read_text(encoding="utf-8")).get("architectures") or [""])[0]
            self._reranker_qwen3 = "Qwen3" in arch
            print(f"[3/3] 加载 Reranker ({RERANKER_MODEL_PATH.name}, {'qwen3-yes/no' if self._reranker_qwen3 else '0/1-token'}) — {r_quant_label}...")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                str(RERANKER_MODEL_PATH), trust_remote_code=True, local_files_only=True
            )
            if RERANKER_QUANT_BITS == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.reranker_model = AutoModelForCausalLM.from_pretrained(
                    str(RERANKER_MODEL_PATH),
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                ).eval()
            elif RERANKER_QUANT_BITS == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                self.reranker_model = AutoModelForCausalLM.from_pretrained(
                    str(RERANKER_MODEL_PATH),
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                ).eval()
            else:
                self.reranker_model = AutoModelForCausalLM.from_pretrained(
                    str(RERANKER_MODEL_PATH),
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                if self.use_gpu:
                    self.reranker_model = self.reranker_model.to(self.device)
                self.reranker_model.eval()
            r_vram = self.reranker_model.get_memory_footprint() / 1024**3
            print(f"      -> VRAM: ~{r_vram:.2f} GB, 量化: {r_quant_label}")
            if self._reranker_qwen3:
                self._reranker_pos_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
                self._reranker_neg_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
            else:
                self._reranker_pos_id = self.reranker_tokenizer.encode("1", add_special_tokens=False)[0]
                self._reranker_neg_id = self.reranker_tokenizer.encode("0", add_special_tokens=False)[0]
        elif self.enable_rerank:
            print(f"[3/3] Reranker 模型路径不存在: {RERANKER_MODEL_PATH}，跳过")
            self.enable_rerank = False
        else:
            print("[3/3] Reranker 已禁用")

        # 打印显存汇总
        if self.use_gpu:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n显存占用: 已分配 {allocated:.2f} GB / 已保留 {reserved:.2f} GB / 总量 {total:.1f} GB")

    def chunk_text(self, text: str, source: str = "") -> List[Dict]:
        """文本分块（按 Markdown 标题，细粒度）"""
        return chunking_by_markdown_headers(text, source)

    def _embed_texts(self, texts: List[str], is_query: bool = False) -> "np.ndarray":
        """批量生成文本向量（按加载时识别的协议自动选择池化方式）

        Args:
            is_query: True 表示查询侧——lasttoken 型模型（Qwen3-Embedding）会自动
                      拼接 Instruct 指令；文档侧不拼指令（与官方用法一致）
        Returns:
            np.ndarray [N, dim] float32（未归一化，调用方负责 normalize）
        """
        if self._emb_lasttoken and is_query:
            texts = [f"Instruct: {EMBED_QUERY_INSTRUCTION}\nQuery:{t}" for t in texts]
        inputs = self.embedder_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self._embedder_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.embedder_model(**inputs)
            hs = outputs.last_hidden_state  # [B, L, H]

            if self._emb_lasttoken:
                # last-token 池化（按 attention_mask 定位真实末 token）
                lengths = inputs["attention_mask"].sum(dim=1) - 1
                emb = hs[torch.arange(hs.size(0), device=hs.device), lengths]
            else:
                # mean 池化（masked），随后可选 2_Dense 线性层
                mask = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)
                emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if self._emb_dense is not None:
                emb = self._emb_dense(emb)
        return emb.float().cpu().numpy()

    def _rerank(self, question: str, candidates: List[Dict], top_k: int = 4) -> List[Dict]:
        """对候选文档重排序，返回 top_k 并附带 rerank_score（0~1，供门控路由）

        双协议（加载时按模型架构自动识别）：
        - Qwen3-Reranker: <Instruct>/<Query>/<Document> prompt，取末位 yes/no
          logit 差做 sigmoid → 0~1
        - mxbai-rerank: query/document + Relevance: prompt，取末位 "0"/"1"
          token 的 softmax 概率 → 0~1
        """
        if not candidates or self.reranker_model is None:
            return candidates[:top_k]

        if self._reranker_qwen3:
            prefix = (
                "<|im_start|>system\n"
                'Judge whether the Document meets the requirements based on the Query '
                'and the Instruct provided. Note that the answer can only be "yes" or "no".'
                "<|im_end|>\n<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            instruction = (
                "Given a customer's question in a livestream e-commerce chat, "
                "judge whether the document section contains the answer"
            )
            prompts = [
                prefix
                + f"<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {c['content']}"
                + suffix
                for c in candidates
            ]
        else:
            # mxbai-rerank prompt（来自 chat_template.jinja）
            def _build_prompt(query: str, document: str) -> str:
                return (
                    "<|im_start|>system\n"
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"query: {query}\n"
                    f"document: {document}\n"
                    "You are a search relevance expert who evaluates how well documents match "
                    "search queries. For each query-document pair, carefully analyze the semantic "
                    "relationship between them, then provide your binary relevance judgment "
                    "(0 for not relevant, 1 for relevant).\n"
                    "Relevance:<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

            prompts = [_build_prompt(question, c["content"]) for c in candidates]

        inputs = self.reranker_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.reranker_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

        scores = []
        for i in range(len(prompts)):
            last_pos = inputs["attention_mask"][i].sum().item() - 1
            last_logits = logits[i, last_pos]
            pos_l = last_logits[self._reranker_pos_id].item()
            neg_l = last_logits[self._reranker_neg_id].item()
            if self._reranker_qwen3:
                scores.append(float(torch.sigmoid(torch.tensor(pos_l - neg_l)).item()))
            else:
                probs = torch.softmax(torch.tensor([neg_l, pos_l]), dim=0)
                scores.append(probs[1].item())

        # 按相关性分数降序，并把分数带回供门控路由使用
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for c, s in ranked[:top_k]:
            item = dict(c)
            item["rerank_score"] = float(s)
            results.append(item)
        return results

    def insert_document(self, text: str, source: str = ""):
        """插入文档（构建索引 — 纯向量，无实体抽取）

        批量 embedding + 内容哈希去重（重复插入/跨文档重复内容自动跳过）
        """
        chunks = self.chunk_text(text, source)
        if not chunks:
            print(f"  警告: {source or '未命名文档'} 未产生任何分块，内容未入库")
            return

        # 内容哈希去重
        new_chunks = []
        for c in chunks:
            h = hashlib.md5(c["content"].encode("utf-8")).hexdigest()
            if h not in self._chunk_hashes:
                self._chunk_hashes.add(h)
                new_chunks.append(c)
        skipped = len(chunks) - len(new_chunks)

        # 批量生成向量嵌入（协议自动：last-token / mean+Dense / CLS）
        embeddings = self._embed_texts([c["content"] for c in new_chunks])

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 向量只存 FAISS 索引，chunks 仅保留元数据（避免内存冗余）
        start_id = len(self.chunks)
        for i, c in enumerate(new_chunks):
            self.chunks.append({
                "id": start_id + i,
                "content": c["content"],
                "source": c.get("source", source),
                "doc_title": c.get("doc_title", ""),
            })

        if self.chunk_index is None:
            dim = embeddings.shape[1]
            self.chunk_index = faiss.IndexFlatIP(dim)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.chunk_index.add(embeddings)

        msg = f"  已插入: {len(new_chunks)} 块"
        if skipped:
            msg += f"（去重跳过 {skipped} 块）"
        print(msg)

    async def query(self, question: str,
                    history: Optional[List[Dict]] = None) -> Dict:
        """查询（纯向量检索 + Rerank + LLM 生成）— 异步锁 + to_thread 防阻塞

        Args:
            history: 对话历史，元素为 {"role": "user"|"assistant", "content": str}，
                     用于指代消解（检索 query 改写）与多轮上下文注入
        """
        async with self._inference_lock:
            return await asyncio.to_thread(self._query_impl, question, history)

    # ── 多轮对话辅助 ──

    # 相对日期触发词：命中才附上代码算好的日期对照（避免每轮注入冗余信息）
    _TIME_HINT_RE = re.compile(
        r"今天|明天|后天|大后天|昨天|星期几|周几|礼拜几|星期[一二三四五六日天]|"
        r"周[一二三四五六日天]|礼拜[一二三四五六日天]|星期末|周末|几号|多少号|几月|日期"
    )
    _WEEK_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

    @classmethod
    def _time_text(cls, question: str) -> str:
        """时间上下文：注入 prompt 让模型直接"抄"日期，不做任何算术。

        实测教训：0.5B 模型无法可靠完成"今天+1天"这类推算——KB 模式下会照
        排班资料（周一~周六）瞎编星期，闲聊模式下偶尔答对但不可靠。因此相对
        日期（昨天/明天/后天/大后天）必须由代码算好，模型只负责照抄。
        """
        def fmt(d: datetime) -> str:
            return f"{d.year}年{d.month}月{d.day}日{cls._WEEK_CN[d.weekday()]}"

        n = datetime.now()
        parts = [f"今天是{fmt(n)}（当前时间 {n.hour:02d}:{n.minute:02d}）"]
        if cls._TIME_HINT_RE.search(question or ""):
            for off, label in ((-1, "昨天"), (1, "明天"), (2, "后天"), (3, "大后天")):
                parts.append(f"{label}是{fmt(n + timedelta(days=off))}")
        return "；".join(parts)

    @staticmethod
    def _history_text(history: Optional[List[Dict]], current: str = "") -> str:
        """把对话历史格式化为 prompt 文本段，无历史时返回空串
        （客户端可能把当前问题也放进 history，跳过末尾重复的当前问题）

        上下文携带：assistant 条目可带 references（该轮命中的参考资料，只取最高分一条），
        渲染为"来源"行，让追问（"一桶多少公斤"）能对准上一轮产品
        """
        if not history:
            return ""
        blocks = []
        for m in history[-HISTORY_MAX_MESSAGES:]:
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if current and content == current:
                continue
            if m.get("role", "user") == "user":
                blocks.append(f"问题：{content}")
            else:
                block = f"答案：{content}"
                ref = next(((r or "").strip() for r in (m.get("references") or [])
                            if (r or "").strip()), "")
                if ref:
                    block += f"\n来源：{ref[:150]}"
                blocks.append(block)
        if not blocks:
            return ""
        return "对话历史：\n\n" + "\n\n".join(blocks) + "\n"

    @staticmethod
    def _retrieval_query(question: str, history: Optional[List[Dict]]) -> str:
        """构建检索 query：把最近用户发言拼接进来做指代消解
        （"是啥牌子的"单独检索会命中错误产品，拼接"有没有大桶的洗洁精"后才能检对）
        """
        if not history:
            return question
        user_msgs = []
        for m in history:
            if m.get("role", "user") == "user":
                c = (m.get("content") or "").strip()
                if c and c != question:  # 跳过当前问题，避免重复拼接
                    user_msgs.append(c)
        if not user_msgs:
            return question
        q = " ".join(user_msgs[-RETRIEVAL_HISTORY_TURNS:] + [question])
        return q[:120]

    def _vector_search(self, query_text: str) -> List[Dict]:
        """向量检索：query embedding → FAISS top-K 候选"""
        query_embedding = self._embed_texts([query_text], is_query=True)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.chunk_index.search(
            query_embedding, min(TOP_K_CANDIDATES, len(self.chunks))
        )

        vector_context = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                vector_context.append({
                    "content": self.chunks[idx]["content"],
                    "score": float(score),
                    "source": self.chunks[idx].get("source", ""),
                    "doc_title": self.chunks[idx].get("doc_title", ""),
                })
        return vector_context

    def _query_impl(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        """查询的同步实现（在 to_thread 中运行，不阻塞事件循环）"""
        print(f"\n问题: {question}")
        t_start = time.perf_counter()

        # 空索引边界：无任何知识库内容时直接走闲聊模式
        if self.chunk_index is None or not self.chunks:
            answer = self.generate_answer(question, [], "chat", history)
            return {
                "question": question,
                "answer": answer,
                "context": [],
                "route_mode": "chat",
            }

        # ── 主检索：只用当前问题本身（新问题绝不被聊天历史带偏）──
        vector_context = self._vector_search(question)
        t_retrieve = time.perf_counter()

        # ── 重排序（路由评分 + chunk 选择）──
        if self.enable_rerank and self.reranker_model is not None:
            route_context = self._rerank(question, vector_context, top_k=RERANK_TOP_K)
        else:
            route_context = vector_context[:RERANK_TOP_K]
        retrieval_source = "当前问题"

        # ── 历史辅助：拼接 query 检索 + 候选池合并 + 选块消解 ──
        # 残句追问（"一件是多少包的"）的语义主体在上文，只用当前问题检索会捞错产品
        # （池外根本没有正确答案，选块消解救不回）：
        # 1) 用拼接 query 再做一次向量检索，结果与当前问题候选池按内容取并集
        #    ——把追问真正需要的产品资料捞回池内
        # 2) 用拼接 query 在合并池上重排选块（只影响选块）
        # 路由评分始终只用当前问题自己的候选池，闲聊不会被历史污染
        selection = None
        if (history and vector_context and self.enable_rerank
                and self.reranker_model is not None):
            retrieval_q = self._retrieval_query(question, history)
            if retrieval_q != question:
                merged_pool = list(vector_context)
                seen = {c["content"] for c in vector_context}
                for c in self._vector_search(retrieval_q):
                    if c["content"] not in seen:
                        merged_pool.append(c)
                        seen.add(c["content"])
                selection = self._rerank(retrieval_q, merged_pool, top_k=RERANK_TOP_K)
                retrieval_source = "当前问题+历史(池合并+选块消解)"
        t_rerank = time.perf_counter()

        # ── 相关性门控路由（单阈值：≥ → kb 注入资料+贪心；< → chat 无资料+采样）──
        route_mode = "chat"
        score_info = ""
        if route_context:
            top1_score = route_context[0].get("rerank_score")
            if top1_score is None:
                route_mode = "kb"
            else:
                score_info = f" (rerank top1: {top1_score:.3f})"
                route_mode = "kb" if top1_score >= RERANK_THRESHOLD else "chat"

        # ── 应用历史辅助结果 ──
        if selection and route_mode == "kb":
            route_context = selection
        print(f"[路由] {route_mode}{score_info} (检索: {retrieval_source})")

        context = route_context

        answer = self.generate_answer(
            question, context if route_mode == "kb" else [], route_mode, history
        )
        t_end = time.perf_counter()

        print(f"[计时] 检索 {(t_retrieve - t_start) * 1000:.0f}ms | "
              f"重排 {(t_rerank - t_retrieve) * 1000:.0f}ms | "
              f"生成 {(t_end - t_rerank) * 1000:.0f}ms | "
              f"总计 {(t_end - t_start) * 1000:.0f}ms")

        return {
            "question": question,
            "answer": answer,
            "context": context if route_mode == "kb" else [],
            "route_mode": route_mode,
        }

    def generate_answer(self, question: str, context: List[Dict],
                        route_mode: str = "kb",
                        history: Optional[List[Dict]] = None) -> str:
        """生成答案（按路由结果使用对应模式的角色提示词和 LLM 参数）

        Args:
            question: 用户问题
            context: 检索上下文（闲聊模式传空列表，不注入资料）
            route_mode: 路由结果，"kb"=知识库问答，"chat"=自由闲聊
            history: 对话历史（多轮上下文与指代消解）
        """
        now_text = self._time_text(question)
        history_text = self._history_text(history, current=question)

        # 统一提示词：一套规则覆盖 KB 与闲聊，接地条款（第 5/6 条）
        # 按"是否提供参考资料"自适应，资料注入与否由路由决定
        has_refs = route_mode == "kb" and bool(context)
        max_len = MAX_ANSWER_LENGTH if has_refs else CHAT_MAX_ANSWER_LENGTH
        rules = SYSTEM_RULES.format(max_answer_length=max_len)

        context_block = ""
        if has_refs:
            context_parts = []
            for i, c in enumerate(context[:RERANK_TOP_K]):
                content = c["content"]
                if len(content) > 2000:
                    content = content[:2000]
                doc_title = c.get("doc_title", "")
                title = f" - {doc_title}" if doc_title else ""
                context_parts.append(f"[来源{i+1}{title}]\n{content}")
            context_block = "参考资料：\n" + "\n\n".join(context_parts) + "\n\n"

        user_content = f"""时间信息：{now_text}
{history_text}
回答规则：
{rules}

{context_block}问题：{question}

回答："""

        messages = [
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": user_content},
        ]

        # 按路由模式选择生成参数：知识库用贪心保证稳定，闲聊用采样保证自然
        if route_mode == "kb":
            gen_kwargs = {"max_new_tokens": LLM_MAX_NEW_TOKENS, "do_sample": LLM_DO_SAMPLE}
            if LLM_DO_SAMPLE:
                gen_kwargs["temperature"] = LLM_TEMPERATURE
                gen_kwargs["top_p"] = LLM_TOP_P
        else:
            gen_kwargs = {
                "max_new_tokens": CHAT_LLM_MAX_NEW_TOKENS,
                "do_sample": CHAT_LLM_DO_SAMPLE,
                "temperature": CHAT_LLM_TEMPERATURE,
                "top_p": CHAT_LLM_TOP_P,
            }
        # 防复读惩罚（抑制 1B 模型偶发的复读/回声退化）
        gen_kwargs["repetition_penalty"] = LLM_REPETITION_PENALTY

        # 1B 模型偶发空白/复读答复：无效则采样重试一次，仍无效用兜底话术
        answer = self._generate_text(messages, gen_kwargs)
        if not self._is_valid_answer(answer, question):
            answer = self._generate_text(
                messages, {**gen_kwargs, "do_sample": True, "temperature": 0.7, "top_p": 0.9}
            )
        if not self._is_valid_answer(answer, question):
            answer = KB_FALLBACK_ANSWER if route_mode == "kb" else CHAT_FALLBACK_ANSWER

        answer = self._smart_truncate(answer, max_len)

        return answer

    def _generate_text(self, messages: List[Dict], gen_kwargs: Dict) -> str:
        """单次 LLM 生成（messages → chat template → generate → decode），返回清洗后的文本"""
        # 调试回显：打印发送给 LLM 的完整消息与生成参数（config.DEBUG_LLM_MESSAGES 控制）
        if DEBUG_LLM_MESSAGES:
            print("\n" + "=" * 64)
            print(f"[LLM 调试] 生成参数: {gen_kwargs}")
            for m in messages:
                print(f"----- [{m['role'].upper()}] -----")
                print(m["content"])
            print("=" * 64)

        text_input = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.llm_tokenizer([text_input], return_tensors="pt").to(self.llm.device)
        # MiniCPM5-1B 不支持 token_type_ids，需要移除
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self.llm.generate(**inputs, **gen_kwargs)

        answer = self.llm_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # 清理可能泄露的对话格式（"问题：..." / "回答：..." 前缀残留）
        for prefix in ["客服回答：", "回答：", "用户："]:
            if prefix in answer:
                parts = answer.split(prefix)
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    break

        return answer

    @staticmethod
    def _is_valid_answer(answer: str, question: str) -> bool:
        """过滤 1B 模型偶发的空白答复和复读式回声"""
        if not answer or len(answer.strip()) < 2:
            return False
        strip_punct = lambda s: re.sub(r"[，。！？、,.!?\s：:；;~～…—-]", "", s)
        return strip_punct(answer) != strip_punct(question)

    @staticmethod
    def _smart_truncate(text: str, max_len: int) -> str:
        """按标点截断，避免把句子截成半截"""
        if len(text) <= max_len:
            return text
        cut = text[:max_len]
        # 回退找最近的句末标点，且不短于限制的一半
        for sep in "。！？；…":
            idx = cut.rfind(sep)
            if idx >= max_len // 2:
                return cut[: idx + 1]
        return cut

    def save(self, path: str):
        """保存索引（仅文本块 + chunk_index）"""
        os.makedirs(path, exist_ok=True)

        if self.chunk_index:
            faiss.write_index(self.chunk_index, os.path.join(path, "chunk_index.faiss"))

        data = {
            "chunks": [{
                "id": c["id"],
                "content": c["content"],
                "source": c.get("source", ""),
                "doc_title": c.get("doc_title", ""),
            } for c in self.chunks],
        }

        with open(os.path.join(path, "data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  索引已保存到: {path}")

    def load(self, path: str):
        """加载索引"""
        faiss_path = os.path.join(path, "chunk_index.faiss")
        bin_path = os.path.join(path, "chunk_index.bin")
        if os.path.exists(faiss_path):
            self.chunk_index = faiss.read_index(faiss_path)
        elif os.path.exists(bin_path):
            self.chunk_index = faiss.read_index(bin_path)
        else:
            print(f"  警告: 未找到 FAISS 索引文件 (chunk_index.faiss / chunk_index.bin)")

        with open(os.path.join(path, "data.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = data["chunks"]

        # 一致性校验 1：chunk 数必须与 FAISS 索引条数一致，错位会导致检索错乱
        if self.chunk_index is not None and len(self.chunks) != self.chunk_index.ntotal:
            raise RuntimeError(
                f"索引数据不一致: data.json 有 {len(self.chunks)} 块，"
                f"FAISS 索引有 {self.chunk_index.ntotal} 条。"
                f"请用完整配对的 data.json + chunk_index.faiss，或重新构建索引"
            )

        # 一致性校验 2：FAISS 索引维度必须与当前 embedder 输出维度一致
        # （换过 embedder 模型或池化方式后旧索引会维度不匹配）
        if self.chunk_index is not None and self.chunk_index.d != self._emb_dim:
            raise RuntimeError(
                f"索引维度不匹配: FAISS 索引为 {self.chunk_index.d} 维，"
                f"当前 embedder ({EMBEDDER_MODEL_PATH.name}) 输出 {self._emb_dim} 维。"
                f"请删除索引目录后重新构建索引"
            )

        # 重建内容哈希集合（供后续插入去重）
        self._chunk_hashes = {
            hashlib.md5(c["content"].encode("utf-8")).hexdigest() for c in self.chunks
        }

        print(f"  已加载 {len(self.chunks)} 个文本块")
