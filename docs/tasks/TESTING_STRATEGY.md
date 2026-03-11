# 🧪 Testing Strategy & Guide

This document outlines the testing hierarchy, execution guidelines, and performance benchmarking for the **RAG System**.

## 1. Test Hierarchy

### 1.1 Automated Suite
*   **Unit Tests (`tests/test_*.py`)**: Isolated logic verification.
*   **Integration Tests (`tests/test_*_integration.py`)**: Subsystem interaction.
*   **Async API Tests (`tests/test_api_modern.py`)**: Serverless FastAPI endpoint verification using `httpx`.
*   **Distributed Tests (`tests/test_distributed_search_v2.py`)**: Multi-node retrieval logic.

### 1.2 Performance Benchmarks
We use dedicated scripts to measure and ensure optimization effectiveness:
*   `scripts/e2e_performance_benchmark.py`: **Final E2E validation** (Parsing → Indexing → Reranking → Answer).
*   `scripts/compare_reranker_perf.py`: Compares Keyword Overlap vs. **FlashRank Semantic Reranking**.
*   `scripts/verify_metadata_opt.py`: Measures **99.8% RAM savings** from reference-based offloading.
*   `scripts/compare_chunking_logic.py`: Validates **Header-Aware** physical separation of chunks.

## 2. Execution Guide

### Standard Test Run
Always run from the project root.
```powershell
# Optimized for Windows PowerShell
python -m pytest tests/
```

### Running New Benchmarks
```powershell
# Full Pipeline Validation
python scripts/e2e_performance_benchmark.py

# Reranker Accuracy Check
python scripts/compare_reranker_perf.py
```

## 3. Stability & Safety
The system undergoes a **Full Stability Pass** after any core change. This involves running the combined suite:
`tests/test_rag_integration.py`, `tests/test_caching_system.py`, `tests/test_api_modern.py`, and `tests/test_distributed_search_v2.py`.

## 4. Best Practices
1.  **Mocking**: Use `patch.object` on modules to ensure mocks are applied correctly regardless of import timing.
2.  **Async**: Use `@pytest.mark.asyncio` for all non-blocking logic tests.
3.  **Clean State**: API tests automatically handle session isolation, but manual scripts should always check `/health` first.
