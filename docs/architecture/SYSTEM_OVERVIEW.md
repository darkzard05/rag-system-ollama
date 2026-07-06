# 🏗️ System Architecture Overview

This document provides a synthesized overview of the **GraphRAG-Ollama** architecture, integrating the latest architectural evolutions (Superpowers) into a cohesive system description.

## 🌟 High-Level Architecture

GraphRAG-Ollama is a high-performance, local RAG system designed for precision and speed. It evolves from a simple linear pipeline to a **Self-Correcting Graph-based Workflow** orchestrated by `LangGraph`.

### The RAG Journey: Data Flow
1. **Ingestion**: PDF $\rightarrow$ `SemanticChunker` (Header-aware) $\rightarrow$ Hybrid Index (FAISS + BM25).
2. **Query Entry**: User Input $\rightarrow$ `preprocess` (Intent Classification & Dynamic Weighting).
3. **Retrieval**: Hybrid Search $\rightarrow$ BGE Semantic Reranking $\rightarrow$ Dynamic Top-K Filtering.
4. **Self-Correction (The Loop)**:
   - **Grading**: BGE Score-based Short-circuit $\rightarrow$ LLM Relevance Validation.
   - **Transformation**: If irrelevant $\rightarrow$ Query Rewriting $\rightarrow$ Re-Retrieval.
5. **Generation**: Context-aware generation with a a-synchronous streaming protocol.
6. **UI Sync**: `SessionStore` $\rightarrow$ `UIBridge` $\rightarrow$ Streamlit UI.

---

## 🚀 Key Architectural Pillars

### 1. Unified Resource Management (`ModelManager`)
To prevent VRAM exhaustion and reduce loading latency, the system uses a centralized resource pool:
- **LRU Caching**: Models (LLM, Embedders, Rerankers) are stored in an `OrderedDict`. Least recently used models are evicted first.
- **Active Pressure Detection**: The manager monitors GPU VRAM and System RAM. If usage exceeds 90%, it proactively triggers eviction to prevent `OutOfMemory` crashes.
- **Single-Instance Policy**: Ensures only one instance of a specific model version exists across the entire application.

### 2. Session Decoupling & SSoT
To solve the "State Fragmentation" problem inherent in Streamlit, the system decouples business logic from the UI:
- **SessionStore (Single Source of Truth)**: A pure Python store that maintains session state independent of the UI framework.
- **UIBridge**: A framework-specific observer that syncs the `SessionStore` to `st.session_state` using a version-based heartbeat mechanism (`session_version`).
- **Thread-Safe Propagation**: Uses a `ContextManager` to safely propagate `session_id` across background worker threads.

### 3. Optimized RAG Pipeline
The pipeline is engineered for "Extreme Speed" without sacrificing precision:
- **Grading Short-circuit**: If the BGE Reranker provides a confidence score $\ge 0.85$, the system skips the costly LLM grading step, reducing latency by 2-3 seconds.
- **Dynamic Top-K**: Instead of a fixed number of documents, the system analyzes the "Score Gap" between the top results to dynamically adjust the reranking window (10-25 docs).
- **1-Pass Pruning**: Redundancy checks are integrated directly into the `SemanticChunker` to avoid duplicate embeddings during indexing.

### 4. Streaming Integrity Protocol
To provide a professional "Ghosting-Free" UI, the system implements a strict streaming protocol:
- **Async Event Dispatch**: Uses `adispatch_custom_event` to send `response_chunk` and `graph_status` events.
- **Buffered Rendering**: Prevents visual flickering by using `st.empty` placeholders and fragment-based updates.

---

## 🛠️ Core Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestrator** | `LangGraph` | Self-correcting workflow management |
| **LLM/Embeddings**| `Ollama` | Local model execution (qwen3, nomic-embed) |
| **Reranker** | `FlashRank` | CPU-optimized semantic reranking (ONNX) |
| **Vector Store** | `FAISS` | High-speed similarity search |
| **Frontend** | `Streamlit` | Interactive UI with fragmented updates |
| **Schema** | `Pydantic v2` | Type-safe configuration and state validation |

---

## 📈 Evolutionary Path
- **v1.x**: Basic RAG pipeline with linear flow.
- **v2.x**: Introduction of self-correction loops and hybrid search.
- **v3.x**: Session decoupling, Unified Resource Management, and BGE-driven short-circuits.
