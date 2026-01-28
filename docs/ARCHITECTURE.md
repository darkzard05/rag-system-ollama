# Project Architecture

This document describes the modular architecture and the logical flow of the GraphRAG-Ollama system, highlighting the latest performance optimizations and UI features.

## 🏗️ Modular Structure

The project follows a package-based structure to ensure high maintainability and clear separation of concerns.

- **`src/core/`**: The brain of the system.
    - `graph_builder.py`: Orchestrates the RAG workflow using LangGraph.
    - `custom_ollama.py`: Implements shared async clients and custom thinking capture.
    - `rag_core.py`: Handles document loading, indexing, and pipeline building.
- **`src/ui/`**: The presentation layer.
    - `ui.py`: Modern Streamlit components featuring grouped controls and real-time status card.
- **`src/common/`**: Utilities and configuration.
    - `utils.py`: Includes **LaTeX normalization** and standardized citation tooltips.
- **`src/services/`**: Optimization and monitoring.
    - `monitoring/`: Real-time tracking of TTFT and token throughput.

## 🧠 Optimized LangGraph Workflow

The generation pipeline is managed as a state-aware graph with advanced concurrency:

1. **`generate_queries`**: Analyzes user input. Simple queries bypass expansion, while complex ones are expanded into 3 targeted search terms.
2. **`retrieve` (Parallel Hybrid Search)**: 
    - **Optimization:** Executes semantic (FAISS) and keyword (BM25) search simultaneously for all queries using `asyncio.gather`.
3. **`generate_response`**: Final streaming inference via **qwen3:4b-instruct**.
    - **Streaming Stability:** Implements an integrity protocol to ensure immediate first-token delivery and prevent duplicate chunks in the UI.

## ⚡ Key Technical Features

### 🧮 Mathematical Support (LaTeX)
The system features a specialized utility, `normalize_latex_delimiters`, which automatically converts various LLM output formats (like `\( ... \)` and `\[ ... \]`) into standardized LaTeX delimiters (`$` and `$$`) compatible with KaTeX rendering in the Streamlit frontend.

### 🎨 Modern UI Orchestration
- **Grouped-Control Navigation:** A professional PDF viewer toolbar that minimizes mouse travel by grouping navigation buttons and integrating a high-precision slider.
- **Real-time Integrity:** The UI loop uses specialized event filtering to maintain a smooth streaming experience even when handling complex reasoning processes (thinking tokens).

### 🚀 Resource Optimization
- **Async Client Pooling:** Reuses Ollama `AsyncClient` instances to eliminate TCP handshake overhead.
- **Sequential Model Loading:** Optimized for local GPU environments to prevent VRAM contention and ensure maximum stability during startup.

## 🛡️ Security
The system employs SHA256 integrity checks and HMAC verification to ensure cached vector stores and model weights have not been tampered with.

---
**Status:** Optimized (v2.2.0) | **Last Updated:** 2026-01-28
