# Project Architecture

This document describes the modular architecture and the logical flow of the RAG (Retrieval-Augmented Generation) system, highlighting the latest performance optimizations.

## 🏗️ Modular Structure

The project follows a package-based structure to ensure high maintainability and clear separation of concerns.

- **`src/core/`**: The brain of the system.
    - `graph_builder.py`: Orchestrates the RAG workflow using LangGraph.
    - `rag_core.py`: Handles document loading, indexing, and pipeline building.
    - `query_optimizer.py`: Complexity-based query routing and expansion.
- **`src/api/`**: The interface layer for external integration.
    - `api_server.py`: FastAPI-based REST and streaming endpoints.
- **`src/services/`**: Background optimization and monitoring tools.
    - `optimization/`: GPU batching, GC tuning, Index optimization (NumPy), and memory management.
    - `monitoring/`: Performance tracking and health checks.
- **`src/security/`**: Multi-layer security implementation.
    - `cache_security.py`: SHA256 integrity and HMAC verification for serialized data.
- **`src/ui/`**: Streamlit-based frontend components with real-time status tracking.

## 🧠 Optimized LangGraph Workflow

The generation pipeline is managed as a state-aware graph with advanced concurrency:

1. **`generate_queries`**: Analyzes user input. Simple queries bypass expansion, while complex ones are expanded into 3 targeted search terms.
2. **`retrieve` (Parallel Hybrid Search)**: 
    - **Optimization:** Executes semantic (FAISS) and keyword (BM25) search simultaneously for all queries using `asyncio.gather`.
    - **Speedup:** Reduces retrieval latency by up to 50% compared to sequential execution.
3. **`rerank_documents`**: Re-evaluates relevance using Cross-Encoders. 
    - **Index Optimization:** Uses **NumPy-based vectorized operations** (Matrix Multiplication) to prune duplicate chunks in O(1) time relative to traditional loops, handling thousands of chunks in milliseconds.
4. **`format_context`**: Structural formatting with precise `[p.X]` page citations.
5. **`generate_response`**: Final streaming inference via **qwen3:4b-instruct**, optimized for instruction following without long thinking loops.

## ⚡ Performance Optimization

### 🚀 Vectorized Document Pruning
The system implements a `DocumentPruner` that utilizes **NumPy matrix multiplication** to calculate cosine similarity between all document vectors at once. This eliminates the O(N²) bottleneck during large document indexing.

### 💾 Efficient Session Management (`doc_pool`)
To minimize memory usage in long conversations, the system implements a **Document Pooling** strategy. Instead of storing full document objects in every chat message, it stores unique document hashes in a central `doc_pool` and references them via IDs, significantly reducing the session state size.

### 🧠 Model Optimization
Defaulting to **qwen3:4b-instruct-2507-q4_K_M** ensures:
- **Zero Thinking Latency:** Instant response generation by bypassing internal CoT loops of base models.
- **Improved Accuracy:** Superior adherence to system prompts and citation rules.

## 🛡️ Security
The system employs SHA256 integrity checks and path validation to ensure cached vector stores and model weights have not been tampered with.

---
**Status:** Optimized (v2.1.0) | **Last Updated:** 2026-01-27