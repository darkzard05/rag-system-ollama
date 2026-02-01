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
- **Vector-Reuse Semantic Chunking:** Achieves ~38% faster processing by reusing embeddings from the semantic split phase directly for FAISS indexing, eliminating redundant model calls.
- **Hardware-Aware UI:** Automatically detects and displays the active compute device (CUDA, MPS, or CPU) in real-time via toast notifications and system logs.
- **Deadlock-Free Session Management:** Implements timeout-based lock acquisition and custom exception handling to prevent system hangs during high-concurrency requests.
- **Integrity-First Streaming:** Utilizes a custom `response_chunk` protocol to ensure zero-duplicate output and immediate first-token delivery (TTFT).

### 🧠 **Intelligent RAG Pipeline**
- **Semantic Boundary Preservation:** Advanced logic ensures meaningful context by maintaining semantic units within user-defined size constraints.
- **LangGraph Orchestration:** Precise control over the retrieval-augmented generation flow for consistent and grounded responses.
- **Hybrid Search:** Combines the strengths of FAISS (Dense) and BM25 (Sparse) retrieval for superior context relevance.

### 🎨 **Refined User Experience**
- **Professional PDF Viewer:** Features a grouped navigation toolbar with precision page sliders and instant rendering.
- **Contextual Status Logging:** Provides detailed, real-time visual feedback for every stage of the document analysis and model loading process.
- **Native Mathematical Support:** Full LaTeX rendering using standardized delimiters for technical and scientific documents.

---

## 🛠️ Getting Started

### 1️⃣ Prerequisites
- **Python 3.11+**
- **Ollama**: Running locally ([ollama.ai](https://ollama.ai))

### 2️⃣ Model Setup
```powershell
# Pull the recommended high-performance model
ollama pull qwen3:4b-instruct-2507-q4_K_M
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

- **Model Selection**: Change default LLM or Embedding models.
- **RAG Parameters**: Adjust chunk size, overlap, and retrieval weights.
- **Security**: Configure RBAC and cache encryption.

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
│   ├── core/               # 🧠 RAG Engine (LangGraph, Chunker, Session)
│   ├── ui/                 # 🎨 Streamlit UI & Components
│   ├── api/                # 🌐 Backend API & Streaming
│   └── common/             # 🛠️ Config, Exceptions & Utils
├── docs/                   # 📚 Technical Documentation (Architecture, Ops, API)
├── reports/                # 📊 Performance & Optimization Audits
├── scripts/                # 🛠️ Maintenance & Debugging Scripts
├── requirements/           # 📦 Dependency management
└── tests/                  # 🧪 Integrity & Verification Tests
```

---

## 📚 Internal Documentation

For more detailed information, please refer to the [Documentation Index](./docs/README.md):
- [Architecture & Protocols](./docs/architecture/ARCHITECTURE.md)
- [Deployment & Security](./docs/ops/DEPLOYMENT.md)
- [API Specifications](./docs/api/API.md)

---

## 📄 License
MIT License - Developed by **darkzard05**.
**Status:** v2.3.0 | **Last Updated:** 2026-01-30
