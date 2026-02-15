# GraphRAG-Ollama

> **A High-Performance, Local Retrieval-Augmented Generation (RAG) Solution with Modern UI.**  
> Optimized for speed and accuracy using `LangGraph` orchestration, local `Ollama` models, and a sleek Streamlit interface.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Model: qwen3:4b-instruct](https://img.shields.io/badge/Model-qwen3:4b--instruct-blueviolet.svg)](https://ollama.com/library/qwen3)
[![Backend: LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Preview

![Application Interface](assets/image1.png)

*GraphRAG-Ollama features a refined sidebar, real-time status logging, and a professional PDF viewer with a grouped-control navigation toolbar.*

---

## ⚡ Key Highlights

### 🚀 **Performance & Stability**
- **IBM Docling Integration:** Delivers high-quality structured Markdown extraction with AI-powered layout analysis, ensuring superior context quality compared to traditional PDF parsers.
- **Ultra-Fast CPU Reranking:** Powered by `FlashRank` (ms-marco-MiniLM-L-12-v2), providing industrial-grade reranking performance directly on CPU with minimal latency.
- **Linear Pipeline Optimization:** Simplified, high-performance RAG path eliminates complex routing overhead, significantly improving Time To First Token (TTFT).
- **Weighted RRF Aggregation:** Advanced hybrid search aggregation (FAISS + BM25) using Weighted Reciprocal Rank Fusion for maximum retrieval precision.
- **Hardware-Aware UI:** Automatically detects and displays active compute devices (CUDA, CPU) and manages VRAM cleanup on system shutdown.

### 🧠 **Intelligent RAG Pipeline**
- **LangGraph Orchestration:** Precise linear control over retrieval, reranking, and generation for consistent and grounded responses.
- **Persistent QA History:** Automatically logs comprehensive conversation history and performance metrics in JSONL format for downstream evaluation and Ragas-based auditing.
- **Integrity-First Streaming:** Utilizes a unified `response_chunk` protocol to ensure zero-duplicate output and immediate delivery of both thoughts and final answers.

### 🎨 **Refined User Experience**
- **Docling-Aware Chunker:** Intelligently splits documents based on Markdown structure (headers, sections) preserved by the Docling engine.
- **Professional PDF Viewer:** Features a grouped navigation toolbar with precision page controls and instant rendering.
- **Real-time Performance Tracker:** Displays detailed metrics including TTFT, TPS, and thinking duration for every interaction.

---

## 🛠️ Getting Started

### 1️⃣ Prerequisites
- **Python 3.11+**
- **Ollama**: Running locally ([ollama.ai](https://ollama.ai))

### 2️⃣ Model Setup
```powershell
# Pull the recommended high-performance models
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull nomic-embed-text
```

### 3️⃣ Installation
```bash
git clone https://github.com/darkzard05/rag-system-ollama.git
cd rag-system-ollama
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate
pip install -r requirements.txt
```

---

## 🐳 Docker Setup (Recommended)

The easiest way to run the entire system is using Docker Compose. This will automatically set up the UI, API, and Ollama server with the required models.

```bash
# Start all services
docker-compose up --build
```

- **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
- **REST API**: [http://localhost:8000](http://localhost:8000)
- **Ollama**: Automatically managed as a container.

---

## 📊 Monitoring & API

If you are using Docker, you can access built-in monitoring and API documentation:

### 📈 Monitoring
- **Grafana**: [http://localhost:3000](http://localhost:3000) (ID/PW: `admin` / `admin`)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)

### 🔌 API Reference
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## ⚙️ Configuration

The system can be customized via `config.yml` or environmental variables in `.env`.

- **Parsing Engine**: Switch between `docling` (default) and `pymupdf` for document extraction.
- **Reranker**: Enable/disable FlashRank and adjust `top_k` and `min_score` parameters.
- **Search Weights**: Tune FAISS and BM25 weights for ensemble retrieval.

---

## 🧪 Testing

Run the full test suite to ensure system integrity:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src
```

---

## 🏗️ Project Structure

```text
rag-system-ollama/
├── src/
│   ├── main.py             # 🏁 Application Entry Point
│   ├── core/               # 🧠 RAG Engine (LangGraph, Docling, Chunker)
│   ├── ui/                 # 🎨 Streamlit UI & Components
│   ├── api/                # 🌐 Backend API & Streaming
│   └── common/             # 🛠️ Config, Exceptions & Utils
├── docs/                   # 📚 Technical Documentation (Architecture, Ops, API)
├── logs/                   # 📝 Persistent QA History & System Logs
├── reports/                # 📊 Performance & Ragas Evaluation Reports
├── scripts/                # 🛠️ Maintenance & Benchmarking Scripts
└── tests/                  # 🧪 Integrity & Verification Tests
```

---

## 📚 Internal Documentation

For more detailed information, please refer to the [Documentation Index](./docs/README.md):
- [Architecture & Protocols](./docs/architecture/STREAMING_INTEGRITY_PROTOCOL.md)
- [Security Implementation](./docs/ops/SECURITY_IMPLEMENTATION.md)
- [Troubleshooting](./docs/ops/TROUBLESHOOTING.md)

---

## 📄 License
MIT License - Developed by **darkzard05**.
**Status:** v3.0.0 | **Last Updated:** 2026-02-15
