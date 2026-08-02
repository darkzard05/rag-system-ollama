import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from cache.engine_cache import EngineCacheManager
from core.rag_core import RAGSystem
from core.session import SessionManager


@pytest.fixture
def mock_resources(monkeypatch):
    """Mock LLM/embedder/retrievers so queries run without real models."""
    session_id = "test-session-123"
    SessionManager.reset()
    SessionManager.init_session(session_id=session_id)
    SessionManager.set("file_hash", "test-hash", session_id=session_id)
    SessionManager.set("pdf_file_path", "test.pdf", session_id=session_id)
    SessionManager.set("last_uploaded_file_name", "test.pdf", session_id=session_id)
    SessionManager.set("llm", MagicMock(), session_id=session_id)
    SessionManager.set("embedder", MagicMock(), session_id=session_id)

    # Mock engine so aquery/astream don't build a real LangGraph
    mock_engine = MagicMock()
    mock_engine.ainvoke = AsyncMock(
        return_value={
            "response": "Mocked Response",
            "thought": "Mocked Thought",
            "relevant_docs": [],
        }
    )
    monkeypatch.setattr(
        EngineCacheManager, "get_engine", staticmethod(lambda sid: mock_engine)
    )

    # Mock resource manager so prepare_query_config doesn't rebuild the pipeline
    mock_rm = MagicMock()
    vector_store = MagicMock()
    vector_store.as_retriever.return_value = MagicMock()
    mock_rm.retrievers.get.return_value = (vector_store, MagicMock())
    monkeypatch.setattr("core.rag_core.get_resource_manager", lambda: mock_rm)
    monkeypatch.setattr("core.pipeline_builder.get_resource_manager", lambda: mock_rm)

    yield mock_engine
    SessionManager.reset()


def test_loop_independence(mock_resources):
    """Two sequential queries (each with its own event loop) must not crash.

    The old loop-ID hot-swapping code used to fail with
    'RuntimeError: Task <...> got Future <...> attached to a different loop'.
    The current design lets LangGraph handle loop changes transparently.
    """
    rag = RAGSystem(session_id="test-session-123")

    def run_query(query):
        return asyncio.run(rag.aquery(query))

    # Loop 1: First query
    res1 = run_query("Hello 1")
    assert res1["response"] == "Mocked Response"

    # Loop 2: Second query (new event loop via asyncio.run)
    res2 = run_query("Hello 2")
    assert res2["response"] == "Mocked Response"
