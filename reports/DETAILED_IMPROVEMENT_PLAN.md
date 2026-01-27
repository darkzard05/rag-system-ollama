# 🛠️ Detailed Improvement Plan & Technical Specification

**Date:** 2026-01-23
**Target:** Refactoring `src/core` and `src/services`
**Status:** Draft

This document provides a concrete technical roadmap for addressing the weaknesses identified in the Code Review.

## 1. Core Logic Refactoring (`src/core/rag_core.py`)

### Problem
The `RAGSystem` class is currently an empty facade. Critical business logic (`_load_pdf_docs`, `build_rag_pipeline`) exists as global functions, leading to:
1.  **Tight Coupling:** Functions rely heavily on `SessionManager` global state.
2.  **Poor Testability:** Difficult to mock internal state for unit tests.
3.  **Low Reusability:** Cannot easily instantiate multiple RAG pipelines (e.g., for different users) in the same process without session collisions.

### Proposed Solution: Object-Oriented Encapsulation
Refactor `RAGSystem` to become the central controller, managing its own state.

#### Blueprint

```python
class RAGSystem:
    def __init__(self, session_id: str = None):
        # State is now instance-local, not global
        self.session_id = session_id or str(uuid.uuid4())
        self.vector_store: Optional[FAISS] = None
        self.retriever: Optional[EnsembleRetriever] = None
        self.qa_chain: Optional[Runnable] = None
        
        # Dependencies
        self.embedder = None
        self.llm = None

    def load_document(self, file_path: str, file_name: str) -> bool:
        """
        Loads PDF, chunks it, and builds vector store.
        Updates self.vector_store directly.
        """
        # Logic from _load_pdf_docs and _load_and_build_retrieval_components
        pass

    def build_pipeline(self, llm_model: str = "llama3"):
        """
        Constructs the LangGraph chain using self.vector_store.
        """
        # Logic from build_graph
        pass

    def query(self, input_text: str) -> Dict:
        """
        Executes the QA chain.
        """
        if not self.qa_chain:
            raise RAGSystemError("Pipeline not initialized")
        return self.qa_chain.invoke({"input": input_text})
```

**Migration Path:**
1.  Update `api_server.py` to instantiate `RAGSystem` per session instead of calling global functions.
2.  Deprecated global functions in `rag_core.py` but keep them as wrappers around `RAGSystem` for temporary backward compatibility.

## 2. Distributed Search Realization (`src/services/distributed/`)

### Problem
`NodeSearchEngine` currently generates fake data in `_init_documents`. This makes the module unusable for real multi-node scenarios.

### Proposed Solution: Interface-based Architecture
Split the implementation into `AbstractNode` and concrete implementations.

#### Blueprint

```python
from abc import ABC, abstractmethod

class AbstractSearchNode(ABC):
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        pass

class LocalMockNode(AbstractSearchNode):
    """Current implementation for testing/demo"""
    def search(self, query):
        # ... existing logic ...
        pass

class HTTPRemoteNode(AbstractSearchNode):
    """Connects to a remote RAG instance"""
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        response = await self.client.post("/api/v1/internal/search", json=query.dict())
        return [SearchResult(**item) for item in response.json()]
```

**Action Items:**
1.  Define `AbstractSearchNode` in `distributed_search.py`.
2.  Rename current `NodeSearchEngine` to `LocalMockNode`.
3.  Implement `DistributedSearchExecutor` to accept a list of `AbstractSearchNode` instances, mixing local and remote nodes.

## 3. Testing Modernization (`tests/`)

### Problem
`tests/test_api_streaming.py` relies on a running server (`requests.get("localhost:8000")`). This is brittle for CI/CD.

### Proposed Solution: Async Client Testing
Use `httpx` and `lifespan` management to test the FastAPI app directly in the test process.

#### Blueprint (`tests/test_api_modern.py`)

```python
import pytest
from httpx import AsyncClient
from src.api.api_server import app

@pytest.mark.asyncio
async def test_streaming_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Mocking the session/upload state might be needed here
        response = await ac.post("/api/v1/stream_query", json={"query": "hello"})
        
        async for line in response.aiter_lines():
            assert line.startswith("event:") or line.startswith("data:")
```

**Action Items:**
1.  Add `httpx` and `pytest-asyncio` to `requirements.txt`.
2.  Create `tests/conftest.py` to handle event loop and app fixtures.

---
**Summary of Benefits:**
*   **Encapsulation:** Reduces side-effects and makes the system easier to reason about.
*   **Scalability:** Prepares the codebase for actual distributed deployment.
*   **Reliability:** Tests become faster and more deterministic.
