"""
Test for P0-Bug: ContextVar snapshot inversion in _start_speculative_generate
causing unauthorized streaming token leak (Event Buffering Bypass).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from common.config import MAX_CONCURRENT_INFERENCE
from core.graph._graph_core import _dispatch_event, _start_speculative_generate
from core.graph.graph_builder import _spec_registry


@pytest.fixture(autouse=True)
def _clear_registry():
    _spec_registry.clear()
    yield
    _spec_registry.clear()


@pytest.mark.asyncio
async def test_speculative_contextvar_binding_order():
    """Verify that _start_speculative_generate correctly sets ContextVar before
    creating the background task via ensure_future, so that events dispatched
    inside the task are buffered in spec.buffer and NOT leaked via adispatch_custom_event
    before adoption.
    """
    mock_dispatch = AsyncMock()

    state = {"input": "test query", "relevant_docs": [], "is_cached": False}
    config = {"configurable": {"thread_id": "test-thread-contextvar"}}

    async def _dummy_generate(st, cfg, *, writer):
        # Dispatch an event inside the speculative task before adoption
        await _dispatch_event(
            "response_chunk", {"content": "leaked_token"}, writer=writer, config=cfg
        )
        return {"response": "done"}

    with (
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph._generate.generate", side_effect=_dummy_generate),
        patch("core.graph._graph_core.adispatch_custom_event", new=mock_dispatch),
    ):
        # Start speculative generate
        started_tid = _start_speculative_generate(state, config, writer=MagicMock())
        assert started_tid == "test-thread-contextvar"

        # Give a moment for the background task to run and call _dispatch_event
        spec = _spec_registry.get("test-thread-contextvar")
        assert spec is not None

        # Await the task completion
        await spec.task

        # Assertions (RED before fix, GREEN after fix):
        # 1. adispatch_custom_event must NOT be called before adoption
        assert mock_dispatch.call_count == 0, (
            f"adispatch_custom_event was called {mock_dispatch.call_count} times, "
            "indicating ContextVar leak / event buffering bypass!"
        )

        # 2. Event must be buffered in spec.buffer
        assert len(spec.buffer) == 1
        assert spec.buffer[0].data == {"content": "leaked_token"}
