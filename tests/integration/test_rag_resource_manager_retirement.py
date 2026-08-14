"""
Plan task 7: RAGResourceManager retirement verification.

- Model switch via ResourceManager.get_*_for_session updates SessionManager
  last_selected_model / last_selected_embedding_model state.
- No RAGResourceManager class remains anywhere in src/.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_OLLAMA_MODEL
from core.resource_manager import ResourceManager


@pytest.mark.asyncio
async def test_get_llm_for_session_updates_last_selected_model():
    """Session-aware wrapper records the selected LLM model in session state."""
    sid = f"test_session_{uuid.uuid4().hex[:8]}"
    manager = ResourceManager()
    mock_llm = MagicMock()

    with patch.object(
        manager, "get_llm", new=AsyncMock(return_value=mock_llm)
    ) as mock_get_llm:
        from core.session import SessionManager

        result = await manager.get_llm_for_session(sid, DEFAULT_OLLAMA_MODEL)

        # Delegates to the existing name-keyed pool with the resolved model name.
        mock_get_llm.assert_awaited_once_with(DEFAULT_OLLAMA_MODEL)
        assert result is mock_llm
        assert (
            SessionManager.get("last_selected_model", session_id=sid)
            == DEFAULT_OLLAMA_MODEL
        )

        SessionManager.delete_session(sid)


@pytest.mark.asyncio
async def test_get_embedder_for_session_updates_last_selected_embedding_model():
    """Session-aware wrapper records the selected embedding model in session state."""
    sid = f"test_session_{uuid.uuid4().hex[:8]}"
    manager = ResourceManager()
    mock_embedder = MagicMock()

    with patch.object(
        manager, "get_embedder", new=AsyncMock(return_value=mock_embedder)
    ) as mock_get_embedder:
        from core.session import SessionManager

        target = AVAILABLE_EMBEDDING_MODELS[0]
        result = await manager.get_embedder_for_session(sid, target)

        # Delegates to the existing name-keyed pool with the resolved model name.
        mock_get_embedder.assert_awaited_once_with(target)
        assert result is mock_embedder
        assert (
            SessionManager.get("last_selected_embedding_model", session_id=sid)
            == target
        )

        SessionManager.delete_session(sid)


def test_rag_resource_manager_removed():
    """RAGResourceManager must no longer exist in the API module."""
    import importlib

    with pytest.raises(AttributeError):
        importlib.import_module("src.api.api_server").RAGResourceManager
