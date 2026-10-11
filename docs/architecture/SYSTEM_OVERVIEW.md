# 🏗️ System Architecture Overview

This document provides a synthesized overview of the **GraphRAG-Ollama** architecture, integrating the latest architectural evolutions (Superpowers) into a cohesive system description.

## 🌟 High-Level Architecture

GraphRAG-Ollama is a high-performance, local RAG system designed for precision and speed. It evolves from a simple linear pipeline to a **Self-Correcting Graph-based Workflow** orchestrated by `LangGraph`.

### The RAG Journey: Data Flow
1. **Ingestion**: PDF $\rightarrow$ `EmbeddingBasedSemanticChunker` (Header-aware) $\rightarrow$ Hybrid Index (FAISS + BM25).
2. **Query Entry**: User Input $\rightarrow$ `preprocess` (Intent Classification & Dynamic Weighting).
3. **Retrieval**: Hybrid Search $\rightarrow$ FlashRank Semantic Reranking $\rightarrow$ Dynamic Top-K Filtering.
4. **Self-Correction (The Loop)**:
   - **Grading**: Rerank Score-based Short-circuit ($\ge$ `min_score_to_skip`) $\rightarrow$ LLM Relevance Validation.
   - **Transformation**: If irrelevant $\rightarrow$ Query Rewriting $\rightarrow$ Re-Retrieval.
5. **Generation**: Context-aware generation with an asynchronous streaming protocol.
6. **UI Sync**: `SessionManager` $\rightarrow$ `UIBridge` $\rightarrow$ Streamlit UI.

---

## 🚀 Key Architectural Pillars

### 1. Unified Resource Management (`ResourceManager`)
To prevent VRAM exhaustion and reduce loading latency, the system uses a centralized resource pool:
- **LRU Caching**: Models (LLM, Embedders, Rerankers) are stored in an `OrderedDict` via `ResourcePool`. Least recently used models are evicted first.
- **Active Pressure Detection**: The manager monitors GPU VRAM and System RAM. If usage exceeds 90%, it proactively triggers eviction to prevent `OutOfMemory` crashes.
- **Model Facade**: `ModelManager` (`src/core/model_loader.py`) is the loading facade on top of `ResourceManager`, providing LLM/embedding/FlashRank access with lazy heavy imports.

### 2. Session Decoupling & SSoT
To solve the "State Fragmentation" problem inherent in Streamlit, the system decouples business logic from the UI:
- **SessionManager (Single Source of Truth)**: A pure Python, thread-safe store (`src/core/session/manager.py`) that maintains session state independent of the UI framework. Includes build-state SSoT, doc-pool copy isolation, and tombstone deletes for safe PDF updates.
- **UIBridge**: A framework-specific observer (`src/ui/bridge.py`) that syncs the `SessionManager` to `st.session_state` using a version-based heartbeat mechanism (`session_version`).
- **Document Analysis SSoT**: The doc-analysis panel (`src/ui/components/doc_analysis/`) is unified on `BuildState` (`src/core/build_state.py`: `initial_build_state`/`apply_event`), surfaced via `st.status` with throttled progress reruns.
- **Thread-Safe Propagation**: Uses a `ContextManager` (`src/core/session/context.py`) to safely propagate `session_id` across background worker threads.

### 3. Optimized RAG Pipeline
The pipeline is engineered for "Extreme Speed" without sacrificing precision:
- **Grading Short-circuit**: If the reranker max score reaches its per-engine threshold (FlashRank `0.70` / semantic bi-encoder `0.45`, plus `min_top_gap_to_skip: 0.05`), the system skips the costly LLM grading step (`rag.prompts.grading.min_score_to_skip`), reducing latency by 2-3 seconds.
- **Retrieval Window**: Stage-1 pool `retriever.search_kwargs.k: 18` matches `dynamic_top_k.max_candidates: 18` (`min: 12`); rerank selection `rerank_top_k: 10`; grading window `grade_top_n: 5` (widened 3→5, F3 fix: citations observed at rank 4–5). Queries shorter than `min_rerank_chars: 6` skip FlashRank and use RRF scores.
- **Lean Index + Coord Memo**: FAISS stores lean metadata (coordinates stripped); `CoordCacheManager` hydrates on demand with an `annotation_memo` TTL table. Hydration drains are bounded (`hydration_drain_timeout_seconds: 30.0`) with parallel extraction and batched hydration.
- **Reference Prewarm**: The reference list prefetches/hydrates non-blocking with order preservation (`src/ui/components/reference_plan.py`), so sources render before citations finalize.
- **1-Pass Pruning**: Redundancy checks are integrated directly into the `EmbeddingBasedSemanticChunker` to avoid duplicate embeddings during indexing.

### 4. Streaming Integrity Protocol
To provide a professional "Ghosting-Free" UI, the system implements a strict streaming protocol:
- **Async Event Dispatch**: `src/core/graph/_generate.py` uses `adispatch_custom_event` to send `response_chunk`, `graph_status`, and mid-stream `citations_add` events (`PartialCitationScanner` in `src/core/graph/_partial_citations.py`).
- **Buffered Rendering**: Prevents visual flickering by using `st.empty` placeholders and fragment-based updates; events are name-tagged at the emitter and consumed generically via `src/ui/components/streaming_core.py` (`merge_citations_add`) and `src/api/stream_pipeline.py` (`StreamChunk.citations_add`).
- **Tail Gates**: First-token/TTFT/post-stream finalize gaps are tracked by `scripts/measure_tail.py`, `scripts/verify_citation_tail.py`, and `scripts/gate_tail.py`.

---

## 🛠️ Core Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestrator** | `LangGraph` | Self-correcting workflow management |
| **LLM/Embeddings**| `Ollama` | Local model execution (qwen3:4b-instruct-2507-q4_K_M, nomic-embed-text-v2-moe) |
| **Reranker** | `FlashRank` | CPU-optimized semantic reranking (ONNX); engine `auto`/`flashrank`/`semantic` |
| **Vector Store** | `FAISS` | High-speed similarity search (HNSW) |
| **Frontend** | `Streamlit` | Interactive UI with fragmented updates |
| **Schema** | `Pydantic v2` | Type-safe configuration and state validation |

---

## 📈 Evolutionary Path
- **v1.x**: Basic RAG pipeline with linear flow.
- **v2.x**: Introduction of self-correction loops and hybrid search.
- **v3.x**: Session decoupling, unified resource management, and score-driven short-circuits. Current focus: performance (GPU batch embedding, cache I/O offload, model preloading) and reasoning extraction (`models.thinking`).
