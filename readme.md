# RAG System with Ollama & LangGraph

> A modular Retrieval-Augmented Generation (RAG) solution powered by Ollama, LangGraph, and Streamlit.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Modular: Packages](https://img.shields.io/badge/Architecture-Modular-orange.svg)]()
[![Backend: LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-informational.svg)]()
[![Tests: 700+](https://img.shields.io/badge/Tests-700+-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏗️ Project Architecture

The system has been refactored into a highly modular package structure to ensure maintainability, scalability, and security.

```text
rag-system-ollama/
├── src/
│   ├── main.py             # 🏁 Main Entry Point (Streamlit UI)
│   ├── core/               # 🧠 Core RAG Logic (Graph, Retrieval, Rerank, Model Loader)
│   ├── api/                # 🔌 API Server & Real-time Handlers (REST, WebSocket)
│   ├── services/           # ⚡ Background Services
│   │   ├── optimization/   #    - AsyncIO, GPU Batching, GC Tuning
│   │   ├── monitoring/     #    - Performance Tracking, Health Checks
│   │   └── distributed/    #    - Cluster Management, Sync
│   ├── security/           # 🛡️ Security Layers (RBAC, Auth, Cache Integrity)
│   ├── common/             # 🛠️ Shared Utilities (Config, Exceptions, Typing)
│   ├── cache/              # 💾 Multi-layer Caching System
│   ├── ui/                 # 🎨 UI Components & Styling
│   └── infra/              # 🏗️ Lifecycle Management (Deployment, Migration, Rollback)
├── tests/                  # 🧪 Comprehensive Test Suites
├── docs/                   # 📚 Detailed Technical Documentation
└── reports/                # 📊 Development & Performance Reports
```

---

## ⚡ Getting Started

### 1️⃣ Prerequisites
- **Python 3.11+**
- **Ollama**: Download and install from [ollama.ai](https://ollama.ai)
- **NVIDIA GPU** (Optional but highly recommended for embedding and inference)

### 2️⃣ Model Setup
Ensure your local Ollama instance is running and pull the required models:

```bash
# Start Ollama service
ollama serve

# Pull required models (Default: qwen3:4b)
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 3️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/darkzard05/rag-system-ollama.git
cd rag-system-ollama

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Environment Variables
cp .env.example .env
```

### 4️⃣ Run the Application

The main entry point is now centralized in `src/main.py`.

```bash
# Start the Streamlit UI
streamlit run src/main.py
```

---

## 🎯 Key Features

### ✨ Engineering Excellence
- **LangGraph Orchestration**: Precise control over the RAG pipeline with state-aware workflows.
- **Smart Optimization**: Automated VRAM detection and batch size calculation for optimal GPU usage.
- **AsyncIO Concurrency**: Parallel retrieval and generation for near-instant responses.
- **Production Resilience**: Integrated Circuit Breakers, Error Recovery Chains, and Deployment Rollback systems.
- **Advanced Security**: HMAC-based cache integrity verification and granular Role-Based Access Control (RBAC).

### 🎨 Logic Flow
```text
📄 PDF Upload → 🔨 Semantic Chunking → 🧮 GPU-Optimized Embedding
      ↓                 ↓                       ↓
🔍 Hybrid Search ← 🔍 Parallel Retrieval ← 🧪 Vector Indexing
      ↓
💡 Streaming LLM Response (via LangGraph) → 💾 Multi-layer Caching
```

---

## 🔌 API & Integration

While the Streamlit UI provides the front-end, the backend is accessible via a modular API layer.

- **REST API**: See `src/api/api_server.py` for endpoints.
- **WebSocket**: Real-time streaming handlers located in `src/api/websocket_handler.py`.
- **Custom Integration**: Use the `SystemIntegration` class in `src/cache/system_integration.py` to embed this RAG system into your own applications.

---

## 🧪 Testing & Quality Assurance

We maintain a rigorous testing standard with over 700+ integrated and unit tests.

```bash
# Run all tests
pytest tests/

# Run specific integration tests
pytest tests/test_rag_integration.py
```

Check the `reports/` directory for historical test results and performance benchmarks.

---

## 📚 Documentation

For deeper technical insights, please refer to the files in the `docs/` directory:
- [API Documentation](./docs/API.md)
- [Architecture Details](./docs/ARCHITECTURE.md)
- [Security Implementation](./docs/SECURITY_IMPLEMENTATION.md)
- [Performance Optimization](./docs/TASK_11_ASYNCIO_OPTIMIZATION.md)

---

## 📄 License

MIT License - Feel free to use and contribute!

---

**Version:** 2.0.0 (Modular Refactor) | **Updated:** 2026-01-22 | **Status:** Stable ✅