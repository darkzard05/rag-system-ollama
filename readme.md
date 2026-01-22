# RAG System with Ollama & LangGraph

> A modular, high-performance Retrieval-Augmented Generation (RAG) solution optimized for local environments.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Modular: Packages](https://img.shields.io/badge/Architecture-Modular-orange.svg)]()
[![Backend: LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-informational.svg)]()
[![Tests: Integrated](https://img.shields.io/badge/Tests-Integrated-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏗️ Project Architecture

The system is organized into specialized packages to ensure maintainability, security, and scalability.

```text
rag-system-ollama/
├── src/
│   ├── main.py             # 🏁 Streamlit UI Entry Point
│   ├── core/               # 🧠 Core RAG Engine (Graph, Retrieval, Optimizer)
│   ├── api/                # 🔌 FastAPI Server (REST & SSE Streaming)
│   ├── services/           # ⚡ Background Optimizers & Monitors
│   ├── security/           # 🛡️ Cache Integrity & Security Layers
│   ├── common/             # 🛠️ Shared Config & Utilities
│   └── cache/              # 💾 Response & Document Caching
├── tests/                  # 🧪 Integration & Unit Test Suites
├── docs/                   # 📚 Technical Documentation
└── reports/                # 📊 Performance & Development Logs
```

---

## ⚡ Getting Started

### 1️⃣ Prerequisites
- **Python 3.11+**
- **Ollama**: Download from [ollama.ai](https://ollama.ai)
- **Ollama Models**: `ollama pull qwen3:4b`

### 2️⃣ Installation
```bash
git clone https://github.com/darkzard05/rag-system-ollama.git
cd rag-system-ollama
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Application

#### **Frontend (Streamlit UI)**
```bash
streamlit run src/main.py
```

#### **Backend (FastAPI Server)**
```bash
uvicorn src.api.api_server:app --host 0.0.0.0 --port 8000
```

---

## 🎯 Key Features

### ✨ Engineering Excellence
- **LangGraph Orchestration**: Precise control over the RAG pipeline with state-aware workflows.
- **Intelligent Query Routing**: Automatically bypasses query expansion for simple questions to minimize latency.
- **Optimized Reranking**: Adaptive filtering and device-aware (CPU/GPU) execution to prevent VRAM bottlenecks.
- **AsyncIO Concurrency**: Parallel retrieval from multiple search engines for near-instant results.
- **Advanced Security**: SHA256 integrity and HMAC verification for all cached artifacts.

### 🎨 Logic Flow
```text
📄 PDF Upload → 🔨 Semantic Chunking → 🧮 Index Optimization
      ↓                 ↓                       ↓
🔍 Hybrid Search ← 🚀 Intelligent Routing ← ⚖️ Adaptive Reranking
      ↓
💡 SSE Response Streaming (via LangGraph) → 🛡️ Cache Integrity
```

---

## 📚 Documentation

Detailed guides for developers:
- [📖 API Reference](./docs/API.md) - Endpoints and integration examples.
- [🏗️ Architecture Details](./docs/ARCHITECTURE.md) - Modular design and logical flow.
- [🛡️ Security Implementation](./docs/SECURITY_IMPLEMENTATION.md) - Cache protection and integrity.
- [⚡ Performance Optimization](./docs/TASK_11_ASYNCIO_OPTIMIZATION.md) - AsyncIO and GPU batching.

---

## 📄 License
MIT License - Developed by darkzard05.

---

**Version:** 2.0.0 | **Updated:** 2026-01-22 | **Status:** Stable ✅
