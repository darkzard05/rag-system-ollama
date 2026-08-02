# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-20
**Commit:** [Current SHA]
**Branch:** [Current Branch]

## OVERVIEW
GraphRAG-Ollama is a high-performance, local Retrieval-Augmented Generation (RAG) solution using LangGraph for orchestration, Ollama for local LLMs, and Streamlit for the UI. It features header-aware semantic chunking and a specialized coordinate cache for efficient PDF highlighting.

## STRUCTURE
```
rag-system-ollama/
├── src/
│   ├── core/       # 🧠 RAG Engine, Semantic Chunker, Graph Builder
│   ├── ui/         # 🎨 Streamlit UI & Components
│   ├── api/        # 🌐 API Server & WebSocket Handlers
│   ├── common/     # ⚙️ Shared Utilities & Config
│   ├── cache/      # 💾 Vector & Coordinate Caching
│   ├── infra/      # 🛠️ Deployment & Recovery Systems
│   ├── security/   # 🛡️ Auth & Encryption
│   └── services/   # 📊 Monitoring & Optimization
├── tests/          # ✅ Unit & Integration Tests
├── scripts/          # 🧪 Benchmarks & Maintenance
└── config.yml      # 📝 Centralized System Configuration
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| RAG Pipeline | `src/core/rag_core.py` | Main orchestration flow |
| Semantic Chunking | `src/core/semantic_chunker.py` | Header-aware splitting logic |
| UI Layout | `src/ui/ui.py` | Main Streamlit page structure |
| API Server | `src/api/api_server.py` | FastAPI/WebSocket implementation |
| System Config | `config.yml` | Model names, prompts, chunk sizes |
| Coordinate Cache | `src/cache/coord_cache.py` | PDF highlight coordinate management |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `RAGOrchestrator` | Class | `src/core/rag_core.py` | Main RAG orchestrator |
| `SemanticChunker` | Class | `src/core/semantic_chunker.py` | Header-aware document splitter |
| `CoordCacheManager` | Class | `src/cache/coord_cache.py` | Manages PDF coordinate caching |
| `SessionManager` | Class | `src/core/session/session_manager.py` | Thread-safe state management |
| `GraphBuilder` | Class | `src/core/graph_builder.py` | Knowledge graph construction |

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
```

## NOTES
- The system is optimized for local execution via Ollama.
- Ensure `nomic-embed-text` and `qwen3` models are pulled before running.

