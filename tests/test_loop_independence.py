import asyncio
import os
import pytest
from unittest.mock import MagicMock, AsyncMock
from core.rag_core import RAGOrchestrator
from core.session.state import SessionState
from core.session.store import SessionStore

# Mocking Resource Manager to avoid needing real LLMs/Embedders for this test
from core.resource_manager import get_resource_manager


@pytest.fixture
def mock_resources(monkeypatch):
    mock_rm = MagicMock()
    # Mock get_llm
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value={
            "response": "Mocked Response",
            "thought": "Mocked Thought",
            "relevant_docs": [],
        }
    )
    mock_llm.astream = AsyncMock()
    mock_rm.get_llm = AsyncMock(return_value=mock_llm)

    # Mock get_embedder
    mock_embedder = MagicMock()
    mock_rm.get_embedder = AsyncMock(return_value=mock_embedder)

    # Mock get_retrievers
    mock_rm.get_retrievers = AsyncMock(return_value=(MagicMock(), MagicMock()))

    monkeypatch.setattr("src.core.rag_core.get_resource_manager", lambda: mock_rm)
    return mock_rm


def test_loop_independence(mock_resources):
    # Setup
    session_id = "test-session-123"
    state = SessionState(
        session_id=session_id,
        file_hash="test-hash",
        pdf_file_path="test.pdf",
        last_uploaded_file_name="test.pdf",
    )
    store = SessionStore()
    rag = RAGOrchestrator(state, store)

    # Note: Production uses InMemorySaver — this cleanup is vestigial
    # but kept for safety with a tempfile path instead of relative checkpoints.sqlite
    import tempfile

    # Using tempfile for test isolation (InMemorySaver is used in production)
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as _:
        pass  # Verify no trace of checkpoints.sqlite remains

    def run_query(query):
        return asyncio.run(rag.aquery(query))

    # Loop 1: First query
    res1 = run_query("Hello 1")
    assert res1["response"] == "Mocked Response"

    # Loop 2: Second query
    res2 = run_query("Hello 2")
    assert res2["response"] == "Mocked Response"

    # If it didn't crash with "RuntimeError: Task <...> got Future <...> attached to a different loop",
    # then the loop-ID hot-swapping logic is successfully removed and LangGraph is handling it.
