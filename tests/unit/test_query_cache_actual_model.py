"""The query-cache key must use the model that actually generated the answer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph._graph_core import _query_cache_key
from core.graph._retrieval import preprocess
from core.session import SessionManager

SID = "query-cache-actual-model"
FILE_HASH = "abc123"
QUERY = "same question"


@pytest.mark.asyncio
async def test_cache_lookup_uses_loaded_model_when_api_overrides_default() -> None:
    SessionManager.set("file_hash", FILE_HASH, session_id=SID)
    SessionManager.set("last_selected_model", "modelA", session_id=SID)
    SessionManager.set("llm_model_name", "modelB", session_id=SID)

    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    noop = AsyncMock(return_value=None)
    try:
        with (
            patch("core.graph._retrieval.QUERY_CACHE_ENABLED", True),
            patch("core.graph._retrieval._ensure_query_cache_embedder", new=noop),
            patch("core.graph._retrieval._precompute_query_embedding", new=noop),
            patch("core.graph._retrieval.get_cache_manager", return_value=cache),
        ):
            await preprocess(
                {"input": QUERY},
                {"configurable": {"session_id": SID, "thread_id": SID}},
                writer=None,
            )

        assert cache.get.call_args.args[0] == _query_cache_key(
            FILE_HASH, QUERY, "modelB"
        )
    finally:
        for key in ("file_hash", "last_selected_model", "llm_model_name"):
            SessionManager.delete(key, session_id=SID)
