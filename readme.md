# RAG System with Ollama & LangGraph

> **A High-Performance, Local Retrieval-Augmented Generation (RAG) Solution.**  
> Optimized for speed and accuracy using `LangGraph` orchestration and local `Ollama` models.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Model: qwen3:4b-instruct](https://img.shields.io/badge/Model-qwen3:4b--instruct-blueviolet.svg)](https://ollama.com/library/qwen3)
[![Backend: LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Preview

![Application Interface](image/image1.png)

*The interface provides a clean chat experience with sidebar controls for document upload and model selection.*

---

## ⚡ Key Highlights

### 🚀 **Performance Optimized**
- **Zero Thinking Time:** Switch to `qwen3:4b-instruct` eliminates the 2-3 minute wait time of standard models, delivering answers in seconds.
- **Async & Parallel:** Uses `AsyncIO` for parallel document retrieval and processing.
- **Device-Aware:** Automatically optimizes embedding and reranking tasks based on available hardware (CPU/GPU).

### 🧠 **Intelligent RAG Pipeline**
- **LangGraph Orchestration:** Precise state management for complex reasoning flows.
- **Hybrid Search:** Combines semantic search (Dense) with keyword search (Sparse/BM25) for best-in-class retrieval.
- **Adaptive Reranking:** Filters irrelevant documents to ensure the LLM receives only high-quality context.

### 🛡️ **Enterprise-Grade Security**
- **Cache Integrity:** Protects cached models and vectors with SHA256 checksums and HMAC verification.
- **Safe Loading:** Prevents unauthorized model loading via strict path validation.

---

## 🛠️ Getting Started

### 1️⃣ Prerequisites
- **Python 3.11+**
- **Ollama**: Download from [ollama.ai](https://ollama.ai)

### 2️⃣ Model Setup (Crucial)
We highly recommend using the **instruct** version of Qwen3 for the best RAG experience (fast response, no long "thinking" loops).

```powershell
# Pull the recommended model
ollama pull qwen3:4b-instruct-2507-q4_K_M

# (Optional) Pull embedding models if not auto-downloaded
# The system will handle this automatically on first run.
```

### 3️⃣ Installation
```bash
git clone https://github.com/darkzard05/rag-system-ollama.git
cd rag-system-ollama

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 4️⃣ Configuration
The system is pre-configured in `config.yml`.
Ensure the default model is set correctly for optimal performance:

```yaml
# config.yml
models:
  default_ollama: "qwen3:4b-instruct-2507-q4_K_M"
```

---

## 🖥️ Usage

### Run the Frontend (Streamlit UI)
The main interface for chatting with your documents.
```bash
streamlit run src/main.py
```

### Run the Backend (API Server)
For integrating RAG capabilities into other applications.
```bash
uvicorn src.api.api_server:app --host 0.0.0.0 --port 8000
```

---

## 🏗️ Project Structure

```text
rag-system-ollama/
├── src/
│   ├── main.py             # 🏁 Streamlit Entry Point
│   ├── core/               # 🧠 RAG Engine (Graph, Retrieval, Models)
│   ├── api/                # 🔌 FastAPI Server
│   ├── services/           # ⚡ Monitoring & Background Services
│   ├── security/           # 🛡️ Security & Cache Verification
│   └── common/             # 🛠️ Config & Utils
├── image/                  # 🖼️ Assets & Screenshots
├── logs/                   # 📝 Application & Performance Logs
├── reports/                # 📊 Benchmarks & Audit Reports
└── tests/                  # 🧪 Test Suite
```

---

## 📚 Documentation & Reports

- **[Model Recommendation Report](./reports/MODEL_SELECTION_RECOMMENDATION.md)**: Why we chose `qwen3:4b-instruct`.
- **[Performance Audit](./reports/PERFORMANCE_AND_QUALITY_AUDIT.md)**: Detailed analysis of system latency and throughput.
- **[API Reference](./docs/API.md)**: Endpoints documentation.

---

## 📄 License
MIT License - Developed by **darkzard05**.

**Status:** Stable (v2.1.0) | **Last Updated:** 2026-01-27