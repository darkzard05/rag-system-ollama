# AGENTS.md - Core RAG Engine

## OVERVIEW
Core RAG engine orchestrating document processing, semantic chunking, hybrid retrieval, and LangGraph-based self-correcting workflows.

## STRUCTURE
| Component | Location | Role |
|-----------|----------|------|
| RAG Orchestrator | `src/core/rag_core.py` | Main `RAGSystem` interface and lifecycle management |
| LangGraph Workflow | `src/core/graph_builder.py` | Self-correcting RAG graph (preprocess, retrieve, grade, rewrite, generate) |
| Semantic Chunking | `src/core/semantic_chunker.py` | Embedding-based semantic text splitting |
| Hybrid Retrieval | `src/core/retriever.py` | BM25 + Vector (FAISS) hybrid search logic |
| Session Management | `src/core/session/` | Thread-safe `SessionState` and `SessionStore` |
| Document Processing | `src/core/document_processor.py` | PDF loading, text extraction, and hash computation |
| Model Management | `src/core/model_loader.py` | LLM and Embedding model lifecycle |
| Resource Pool | `src/core/resource_pool.py` | Global registry for vector stores and retrievers |
| Reranking | `src/core/reranker.py` | Distributed semantic reranking |
| Chunking Utils | `src/core/chunking.py` | Low-level document splitting utilities |

## WHERE TO LOOK
- **Modify RAG Pipeline**: `src/core/rag_core.py`
- **Update Workflow Logic**: `src/core/graph_builder.py`
- **Adjust Chunking Strategy**: `src/core/semantic_chunker.py`
- **Tweak Retrieval/Reranking**: `src/core/retriever.py` or `src/core/reranker.py`
- **Manage Session State**: `src/core/session/state.py`

## CONVENTIONS
- **Statefulness**: Always use `SessionState` and `SessionStore` for stateful operations.
- **Async First**: All core RAG operations are asynchronous; ensure proper `await` usage.
- **Entry Point**: Use `RAGSystem` as the primary interface for all RAG-related tasks.
- **Typing**: Mandatory type annotations for all signatures.

## ANTI-PATTERNS
- **Bypassing Session**: Do not manage session state manually; use the `session/` module.
- **Hardcoded Models**: Never hardcode model names; use `ModelManager`.
- **UI Logic in Core**: Keep all business logic in `src/core/`; `src/ui/` is for presentation only.
- **Bare Exceptions**: Never use `except: pass`. Use specific exceptions from `src/common/exceptions.py`.
