# GraphRAG-Ollama

> **A High-Performance, Local Retrieval-Augmented Generation (RAG) Solution with Modern UI.**  
> Optimized for speed and accuracy using `LangGraph` orchestration, local `Ollama` models, and a sleek Streamlit interface.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Model: qwen3:4b-instruct](https://img.shields.io/badge/Model-qwen3:4b--instruct-blueviolet.svg)](https://ollama.com/library/qwen3)
[![Backend: LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Preview

![Application Interface](image/image1.png)

*GraphRAG-Ollama features a refined sidebar, real-time status logging, and a professional PDF viewer with a grouped-control navigation toolbar.*

---

## ⚡ Key Highlights

### 🚀 **Performance & Stability**
- **Optimized Streaming:** Immediate first-token delivery with an improved integrity protocol to prevent duplicate outputs.
- **Zero Thinking Latency:** Pre-configured for `qwen3:4b-instruct`, eliminating long reasoning loops while maintaining high accuracy.
- **Async Resource Management:** Shared Ollama clients and intelligent model loading strategies to maximize GPU efficiency.

### 🧠 **Intelligent RAG Pipeline**
- **Modern Orchestration:** Powered by `LangGraph` for precise control over retrieval and generation flows.
- **Mathematical Support:** Native LaTeX rendering for complex formulas using standardized $ and $$ delimiters.
- **Hybrid Search & Reranking:** Combines FAISS (Dense) and BM25 (Sparse) with optional reranking for superior context relevance.

### 🎨 **Refined User Experience**
- **Grouped Navigation Toolbar:** A professional PDF viewer toolbar with grouped "Prev/Next" buttons and a precision page slider.
- **Modern Sidebar:** Compact design using `st.popover` for advanced settings and custom status logs.
- **Real-time Feedback:** Integrated `st.toast` notifications and dynamic status updates for long-running operations.

---

## 🛠️ Getting Started

### 1️⃣ Prerequisites
- **Python 3.11+**
- **Ollama**: Download from [ollama.ai](https://ollama.ai)

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

## 🖥️ Usage

### Run the Frontend (Streamlit UI)
```bash
streamlit run src/main.py
```

### Run the Backend (FastAPI Server)
```bash
uvicorn src.api.api_server:app --host 0.0.0.0 --port 8000
```

---

## 🏗️ Project Structure

```text
rag-system-ollama/
├── src/
│   ├── main.py             # 🏁 Streamlit Entry Point
│   ├── core/               # 🧠 RAG Engine (LangGraph, Custom Ollama)
│   ├── ui/                 # 🎨 Modern UI Components
│   └── common/             # 🛠️ Math Utils & Config
├── docs/                   # 📚 Technical Documentation
├── reports/                # 📊 Optimization & Audit Reports
└── tests/                  # 🧪 Integrity & Flow Tests
```

---

## 📄 License
MIT License - Developed by **darkzard05**.
**Status:** v2.2.0 | **Last Updated:** 2026-01-28
