# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-09
**Commit:** a0778fc
**Branch:** main

## OVERVIEW
GraphRAG-Ollama is a high-performance, local Retrieval-Augmented Generation (RAG) solution using LangGraph for orchestration, Ollama for local LLMs, and Streamlit for the UI. It features header-aware semantic chunking and a specialized coordinate cache for efficient PDF highlighting.

## STRUCTURE
```
rag-system-ollama/
├── src/
│   ├── core/       # 🧠 RAG Engine, Semantic Chunker, Graph Builder, Session
│   ├── ui/         # 🎨 Streamlit UI (components/, styles/) & Bridge
│   ├── api/        # 🌐 API Server & SSE Streaming
│   ├── common/     # ⚙️ Shared Utilities & Config
│   ├── cache/      # 💾 Vector & Coordinate Caching
│   ├── infra/      # 🛠️ System Notifier
│   ├── security/   # 🛡️ Auth & Cache Security
│   └── services/   # 📊 Monitoring & Optimization
├── tests/          # ✅ Unit, Integration, Security & E2E Tests
├── scripts/          # 🧪 Benchmarks, Evaluation & Maintenance
└── config.yml      # 📝 Centralized System Configuration
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| RAG Pipeline | `src/core/rag_core.py` | `RAGSystem` orchestration flow |
| Semantic Chunking | `src/core/chunking.py` | `split_documents()` orchestration entry |
| Pure Splitter | `src/core/semantic_chunker.py` | `EmbeddingBasedSemanticChunker` (header-aware) |
| Graph Workflow | `src/core/graph_builder.py` | LangGraph nodes + `build_graph()` |
| Session State | `src/core/session/manager.py` | `SessionManager` (SSoT, thread-safe) |
| UI Bridge | `src/ui/bridge.py` | Syncs `SessionManager` ↔ `st.session_state` |
| UI Layout | `src/ui/ui.py` | Main Streamlit page structure |
| API Server | `src/api/api_server.py` | FastAPI/SSE implementation |
| System Config | `config.yml` | Model names, prompts, chunk sizes |
| Coordinate Cache | `src/cache/coord_cache.py` | PDF highlight coordinate management |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `RAGSystem` | Class | `src/core/rag_core.py` | Main RAG orchestrator |
| `PipelineBuilder` | Class | `src/core/pipeline_builder.py` | Pipeline build/caching service |
| `EmbeddingBasedSemanticChunker` | Class | `src/core/semantic_chunker.py` | Header-aware document splitter |
| `split_documents` | Function | `src/core/chunking.py` | Chunking orchestration entry |
| `build_graph` | Function | `src/core/graph_builder.py` | LangGraph state graph construction |
| `CoordCacheManager` | Class | `src/cache/coord_cache.py` | Manages PDF coordinate caching |
| `SessionManager` | Class | `src/core/session/manager.py` | Thread-safe state management (import: `from core.session import SessionManager`) |
| `ContextManager` | Class | `src/core/session/context.py` | Thread-safety context for background tasks |
| `UIBridge` | Class | `src/ui/bridge.py` | Syncs `SessionManager` ↔ Streamlit state |
| `ResourceManager` | Class | `src/core/resource_manager.py` | LRU model/retriever pools with VRAM pressure eviction |
| `ModelManager` | Class | `src/core/model_loader.py` | LLM/embedding/flashrank facade |

## CONVENTIONS
- **Python 3.10+**: Use `X | Y` for unions.
- **Formatting**: Strictly `ruff` (Double quotes, 88 chars).
- **Typing**: Mandatory type annotations for all signatures (`mypy`).
- **Imports**: Sorted via `ruff`. Lazy imports for heavy modules (`torch`, `fitz`).

## ANTI-PATTERNS (THIS PROJECT)
- **No Bare Excepts**: Never use `except: pass`. Use specific exceptions from `src/common/exceptions.py`.
- **No Hardcoded Secrets**: All credentials must be in `.env` or `config.yml`.
- **No UI Logic in Core**: Keep `src/ui/` separate from business logic in `src/core/`.

## UNIQUE STYLES
- **Reference-based Metadata Offloading**: Coordinates are stored in a separate cache to keep FAISS indices lean.
- **Header-Aware Chunking**: Chunks are grouped by Markdown headers to preserve structural context.

## COMMANDS
```bash
# Run Application
streamlit run src/main.py

# Quality Assurance
ruff check .
ruff format .
mypy src
pytest tests/unit
python scripts/test_full_pipeline.py

# CI also enforces (see .github/workflows/ci.yml)
bandit -r src/ -ll
pip-audit -r requirements.txt --skip-editable
pytest --cov=src --cov-fail-under=55 tests/unit
```

## NOTES
- The system is optimized for local execution via Ollama.
- Pull the configured models before running: `qwen3:4b-instruct-2507-q4_K_M` (LLM) and `nomic-embed-text-v2-moe` (default embedding) — both set in `config.yml`.
- Test coverage gate is ≥55%; `scripts/maintenance/verify_integrity.py` runs a full style/typing/RAG-logic check.

