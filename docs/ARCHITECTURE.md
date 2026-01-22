# Project Architecture

This document describes the modular architecture and the logical flow of the RAG (Retrieval-Augmented Generation) system.

## 🏗️ Modular Structure

The project follows a package-based structure to ensure high maintainability and clear separation of concerns.

- **`src/core/`**: The brain of the system.
    - `graph_builder.py`: Orchestrates the RAG workflow using LangGraph.
    - `rag_core.py`: Handles document loading, indexing, and pipeline building.
    - `query_optimizer.py`: Complexity-based query routing and expansion.
- **`src/api/`**: The interface layer for external integration.
    - `api_server.py`: FastAPI-based REST and streaming endpoints.
- **`src/services/`**: Background optimization and monitoring tools.
    - `optimization/`: GPU batching, GC tuning, and memory optimization.
    - `monitoring/`: Performance tracking and health checks.
- **`src/security/`**: Multi-layer security implementation.
    - `cache_security.py`: SHA256 integrity and HMAC verification for serialized data.
- **`src/ui/`**: Streamlit-based frontend components.

## 🧠 LangGraph Workflow

The generation pipeline is managed as a state-aware graph:

1. **`generate_queries`**: Analyzes user input. Simple queries bypass expansion for speed, while complex ones are expanded into multiple search terms.
2. **`retrieve`**: Parallel execution of semantic (Vector) and keyword (BM25) search.
3. **`rerank_documents`**: Re-evaluates the relevance of retrieved chunks using a Cross-Encoder model. Adaptive filtering is applied to balance speed and quality.
4. **`format_context`**: Structural formatting of context with precise page citations.
5. **`generate_response`**: Final streaming inference via Ollama.

## ⚡ Performance Optimization

- **Multi-layer Caching**: Combining exact-match (L1) and semantic similarity (L2) caching.
- **Batch Optimizer**: Real-time VRAM detection to adjust GPU workload.
- **Adaptive GC**: Context-aware garbage collection to prevent stuttering during streaming.

## 🛡️ Security

The system employs rigorous cache validation to prevent Pickle-based deserialization attacks, ensuring that local data remains untampered.
