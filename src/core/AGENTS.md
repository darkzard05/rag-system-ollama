# AGENTS.md - Core RAG Engine

## OVERVIEW
Core RAG engine orchestrating document processing, semantic chunking, hybrid retrieval, and LangGraph-based self-correcting workflows.

## STRUCTURE
| Component | Location | Role |
|-----------|----------|------|
| RAG Orchestrator | `src/core/rag_core.py` | Main `RAGSystem` interface and lifecycle management |
| LangGraph Workflow | `src/core/graph_builder.py` | Self-correcting RAG graph (preprocess, retrieve, grade, rewrite, generate) |
| Semantic Chunking | `src/core/semantic_chunker.py` | Embedding-based semantic text splitting |
| Hybrid Retrieval | `src/core/retriever_factory.py` | BM25 + Vector (FAISS) hybrid search logic |
| Session Management | `src/core/session/` | Thread-safe `SessionManager` |
| Document Processing | `src/core/document_processor.py` | PDF loading, text extraction, and hash computation |
| Model Management | `src/core/model_loader.py` | LLM and Embedding model lifecycle |
| Resource Pool | `src/core/resource_manager.py` | Global registry for vector stores, retrievers, and inference locks |
| Reranking | `src/core/async_reranker.py` | `AsyncCrossEncoderReranker` (FlashRank cross-encoder with `engine: auto`), falls back to bi-encoder `AsyncSemanticReranker` on failure |
| Graph State Reducers | `src/api/schemas.py` | `reset_or_append` (`search_queries`) / `reset_or_add` (`retry_count`) turn-boundary reset reducers |
| Streaming Consumption | `src/ui/components/streaming.py` | `consume_stream_into_message` persists streamed answers synchronously (no polling fragment) |
| Chunking Utils | `src/core/chunking.py` | Low-level document splitting utilities |

## WHERE TO LOOK
- **Modify RAG Pipeline**: `src/core/rag_core.py`
- **Update Workflow Logic**: `src/core/graph_builder.py`
- **Adjust Chunking Strategy**: `src/core/semantic_chunker.py`
- **Tweak Retrieval/Reranking**: `src/core/retriever_factory.py` or `src/core/async_reranker.py`
- **Manage Session State**: `src/core/session/manager.py`

## CONVENTIONS
- **Statefulness**: Always use `SessionManager` for stateful operations.
- **Async First**: All core RAG operations are asynchronous; ensure proper `await` usage.
- **Entry Point**: Use `RAGSystem` as the primary interface for all RAG-related tasks.
- **Typing**: Mandatory type annotations for all signatures.

## ANTI-PATTERNS
- **Bypassing Session**: Do not manage session state manually; use the `session/` module.
- **Hardcoded Models**: Never hardcode model names; use `ModelManager`.
- **UI Logic in Core**: Keep all business logic in `src/core/`; `src/ui/` is for presentation only.
- **Bare Exceptions**: Never use `except: pass`. Use specific exceptions from `src/common/exceptions.py`.
