from __future__ import annotations

import pytest

from common.exceptions import VectorStoreError
from core.rag_core import RAGSystem
from core.resource_manager import get_resource_manager
from core.session import SessionManager


@pytest.mark.asyncio
async def test_astream_setup_failure_pin_leak():
    session_id = "test_f1_session"
    SessionManager.init_session(session_id=session_id)
    SessionManager.set("file_hash", "dummy_file_hash_123", session_id=session_id)

    resource_manager = get_resource_manager()
    resource_manager.reset()

    from unittest.mock import MagicMock

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()

    # Register retriever in resource manager so pin works
    await resource_manager.register_retrievers(
        "dummy_file_hash_123", mock_vs, MagicMock()
    )

    rag_system = RAGSystem(session_id=session_id)

    async def mock_get_rag_engine():
        return None  # Triggers VectorStoreError during astream setup

    rag_system._get_rag_engine = mock_get_rag_engine

    with pytest.raises(VectorStoreError):
        await rag_system.astream("test query")

    # Verification: pin must not leak when setup fails before consumer is created
    assert not resource_manager.retrievers.is_pinned("dummy_file_hash_123"), (
        "Retriever pin leaked when astream setup failed before consumer creation!"
    )
