# 🚀 Performance & Quality Audit Report

**Date:** 2026-01-23
**Scope:** Model Config, RAG Pipeline, Optimization Strategies
**Auditor:** Gemini Agent

## 1. Executive Summary
The system architecture is highly optimized for concurrency (`asyncio`) and resource management (`GC tuning`). However, the default configuration for the LLM context window (`2048`) and the aggressive query expansion strategy may bottleneck performance on local hardware running `qwen3:4b`.

## 2. Detailed Analysis

### 2.1 Model Configuration (`src/common/config.py`)
*   **Current State:**
    *   Model: `qwen3:4b` (Good balance for local RAG)
    *   Context Window (`num_ctx`): **2048**
    *   Timeout: 900s
*   **Impact:**
    *   **Quality:** 2048 tokens is often insufficient for RAG. If retrieved chunks + system prompt exceed this, the LLM will truncate the context, leading to hallucinations or "I don't know" answers.
*   **Recommendation:** Increase `DEFAULT_OLLAMA_NUM_CTX` to **4096** or **8192** (if VRAM permits).

### 2.2 RAG Pipeline Strategy (`src/core/graph_builder.py`)
*   **Query Expansion:**
    *   Currently enabled by default.
    *   **Impact:** Adds an extra LLM round-trip before retrieval. On a local 4B model, this adds ~1-3 seconds of latency.
    *   **Recommendation:** Make Query Expansion **opt-in** or strictly limit it to ambiguous queries only (tune `RAGQueryOptimizer` threshold).
*   **Reranking:**
    *   Fetches `max_rerank_docs=12` and selects top `k=6`.
    *   **Impact:** Good balance. `Cross-Encoder` reranking is expensive. 12 docs is a safe upper limit for CPU-based reranking.

### 2.3 Optimization Modules (`src/services/optimization/`)
*   **Vector DB Optimizer:**
    *   Implements `BatchIndexer` and `ParallelSearcher`.
    *   **Verdict:** Excellent. Parallel searching significantly reduces the latency of hybrid search (BM25 + FAISS).
*   **GC Tuner:**
    *   Contextual GC is enabled.
    *   **Verdict:** Essential for preventing stutter during streaming generation.

### 2.4 Prompt Engineering
*   **System Prompt:** `QA_SYSTEM_PROMPT` (Loaded from config).
*   **Observation:** Ensure the prompt explicitly instructs the model to "think step-by-step" (Chain-of-Thought) to improve reasoning quality, even if it slightly increases generation time.

## 3. Actionable Recommendations

### ⚡ To Improve Speed (Latency)
1.  **Disable Query Expansion** by default for simple queries.
2.  **Reduce `top_k`** for Reranker from 12 to 8 if CPU usage is high.
3.  **Stream Optimization:** Ensure `adispatch_custom_event` is not buffering tokens.

### 💎 To Improve Quality (Accuracy)
1.  **Increase Context Window:** Set `OLLAMA_NUM_CTX=4096` in `config.yml` or env vars.
2.  **Hybrid Search Weights:** Verify `EnsembleRetriever` weights (currently loaded from config). Ideally `0.5/0.5` or `0.4/0.6` (Sparse/Dense) works best for general domains.
3.  **Chunk Merging:** The current `_merge_consecutive_chunks` logic is good but could be more aggressive to provide larger contiguous context to the LLM.

## 4. Conclusion
The system requires **configuration tuning** rather than code refactoring. The underlying engine is robust. Focusing on the Context Window size and Query Expansion overhead will yield the biggest immediate gains.
