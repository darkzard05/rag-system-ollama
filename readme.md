# GraphRAG-Ollama

> **A High-Performance, Local Retrieval-Augmented Generation (RAG) Solution with Modern UI.**  
> Optimized for extreme speed and precision using `LangGraph` orchestration, local `Ollama` models, and a sleek Streamlit interface.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![Model: qwen3:4b-instruct](https://img.shields.io/badge/Model-qwen3:4b--instruct-blueviolet.svg)](https://ollama.com/library/qwen3)
[![Backend: LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Preview

![Application Interface](assets/image1.png)

*GraphRAG-Ollama features a refined sidebar, real-time status logging, and a professional PDF viewer with a grouped-control navigation toolbar.*

---

## ⚡ Key Highlights (v3.3.0 Updated)

### 🚀 **Extreme Performance & Efficiency**
- **Header-Aware Semantic Chunking:** Beyond simple text splitting, our system respects Markdown headers (`#`, `##`). It prevents context contamination between sections and injects structural metadata (`current_section`) into every chunk.
- **99.8% Metadata Optimization:** Implements **Reference-based Metadata Offloading**. Word coordinates for highlighting are stored in a dedicated side-cache (`CoordCacheManager`), reducing FAISS index RAM usage by over 99% while maintaining sub-millisecond hydration during retrieval.
- **FlashRank Semantic Reranking:** Integrated `FlashRank` (v0.2.0) with ONNX runtime. Re-evaluates search results using a Cross-Encoder model on CPU, achieving **2x higher accuracy (P@1)** with minimal latency.

### 🧠 **Intelligent Reasoning & Citations**
- **Configurable Prompt System:** All RAG prompts (`grading`, `rewriting`, `QA`) are externalized in `config.yml`. They use generalized instructions to handle complex model names and technical terms without hardcoding.
- **Section-Aware Citations:** Implements an advanced citation processor that extracts section names and matches them with document metadata. Provides **Interactive Citation Badges** with sub-second preview tooltips.
- **Relevant Entity Extraction:** The LLM is strictly instructed to extract key technical entities during the grading phase, ensuring transparency in its reasoning process.

### 🛡️ **Reliability & Integrity**
- **Multi-Layer Unit Testing:** 13+ new unit tests covering Graph Flow, RAG Orchestration, Document Processing, and Advanced Citations.
- **100% Integrity Pass:** Integrated verification system (`verify_integrity.py`) ensures code style, typing, and RAG logic are always production-ready.
- **Ghosting-Free UI:** Advanced `st.fragment` and `st.empty` placeholder management prevents visual glitches during real-time streaming.

---

## 🛠️ Tech Stack
<!-- TECH_STACK_START -->
- **Streamlit**: 1.54.0
- **LangChain**: 0.3.x (latest)
- **LangGraph**: 0.2.x (latest)
- **FlashRank**: 0.2.0 (ONNX Optimized)
- **PyMuPDF4LLM**: 0.3.4+
- **Ollama**: 0.6.x
- **FAISS**: CPU Optimized
<!-- TECH_STACK_END -->

---

## 🏗️ Project Structure
<!-- TREE_START -->
```text
rag-system-ollama/
├── src/
│   ├── cache/          # 💾 CoordCache, VectorCache
│   ├── common/         # ⚙️ Config (config.yml integration), Utils
│   ├── core/           # 🧠 LangGraph Engine, Semantic Chunker (Header-aware)
│   ├── infra/          # 🛠️ Deployment & Error Recovery
│   ├── main.py         # 🏁 Streamlit Entry Point
│   ├── services/       # 📊 Performance Monitoring
│   └── ui/             # 🎨 Streaming UI Components
├── config.yml          # 📝 Centralized RAG & Model Configuration
├── scripts/            # 🧪 Benchmarks & Maintenance
└── tests/              # ✅ Comprehensive Unit (Pytest) & Integration Tests
```
<!-- TREE_END -->

---

## 🚀 Getting Started

### 1️⃣ Model Setup
```powershell
# Pull the recommended models
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull nomic-embed-text
```

### 2️⃣ Configuration
Customize your RAG behavior in `config.yml` (e.g., grading instructions, chunk sizes, model names).

### 3️⃣ Running the App
```bash
# Optimized for Windows environments
streamlit run src/main.py
```

---

## 🧪 Testing & Verification

We maintain a strict **Zero-Error Policy**. Run the automated verification suite:
```powershell
# Run all unit tests
pytest tests/unit

# Run full pipeline integration test
python scripts/test_full_pipeline.py

# Verify section metadata extraction
python scripts/verify_section_metadata.py
```

---

## 📄 License
MIT License - Developed by **darkzard05**.
**Status:** v3.3.0 | **Last Updated:** 2026-03-12
