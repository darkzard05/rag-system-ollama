# 🧪 Testing Strategy & Guide

This document outlines the testing hierarchy, execution guidelines, and performance benchmarking for the **GraphRAG-Ollama** system.

## 1. Test Hierarchy

### 1.1 Multi-Layer Unit Tests (`tests/unit/`)
We implement a deep unit testing strategy to ensure each core component functions reliably in isolation:
*   **Graph Flow (`test_graph_flow.py`)**: Validates the **LangGraph** orchestration. Mocks LLM and retrievers to verify state transitions (General intent vs. RAG intent, retry loops, and cache hit paths).
*   **RAG Orchestration (`test_rag_pipeline.py`)**: Tests the `RAGSystem` class lifecycle including pipeline building, indexing coordination, and query routing.
*   **Document Processor (`test_document_processor.py`)**: Validates PDF layout diagnosis (1-column vs. 2-column) and intelligent section filtering (TOC/Reference removal).
*   **Advanced Citations (`test_citation_advanced.py`)**: Ensures citation normalization and section-aware matching work across complex patterns like `[섹션: ..., p.X]`.
*   **Semantic Chunking (`test_semantic_chunking.py`)**: Verifies that Markdown headers are preserved and not merged into body text.

### 1.2 Integration & API Tests
*   **E2E Integration (`tests/integration/`)**: Subsystem interaction tests using local model mocks.
*   **Async API Tests (`tests/test_api_modern.py`)**: FastAPI endpoint verification using `httpx`.

### 1.3 Live E2E & Benchmarks
Dedicated scripts to validate the real pipeline and measure optimization effectiveness:
*   `scripts/verification/verify_full_pipeline.py`: **Core E2E validation** using a real PDF (`2201.07520v1.pdf`), live Ollama, build-stage events, golden queries, and streaming citations.
*   `scripts/benchmarks/compare_chunking_methods.py`: Recursive vs config-driven (semantic/chonkie) chunking on the real PDF.
*   `scripts/evaluation/quick_eval.py`: LLM-judged answer quality on `testset_2201.csv`.

## 2. Execution Guide

### Standard Test Run
Always run from the project root. We recommend running unit tests frequently during development.
```powershell
# Run all core unit tests
pytest tests/unit

# Run with coverage report
pytest --cov=src tests/unit
```

### Full Pipeline Validation (Real PDF)
Run this after making changes to the LLM prompts, chunking engine, or citation stream.
```powershell
# full: build + golden set + streaming citations (default)
python scripts/verification/verify_full_pipeline.py

# fast: build + two in-document queries
python scripts/verification/verify_full_pipeline.py --mode smoke
```

## 3. CI/CD & Stability Pass
The system undergoes a **Full Stability Pass** before any release. This involves:
1. `pytest tests/unit`
2. `python scripts/maintenance/verify_integrity.py`
3. `python scripts/verification/verify_full_pipeline.py`

## 4. Best Practices
1.  **Mocking (Astream Support)**: When testing LangGraph nodes, ensure you mock both `ainvoke` and `astream` (using asynchronous generators) to match the real application behavior.
2.  **State Reducers**: Always verify that the `GraphState` reducers (e.g., `operator.add` for search queries) are accumulating state correctly during loops.
3.  **Prompt Isolation**: Test prompts by injecting mock LLM responses that follow the expected structure (`GradeResponse`, `RewriteResponse`).
