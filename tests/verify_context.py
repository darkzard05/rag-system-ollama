import asyncio

import pytest

from core.session.context import ContextManager


@pytest.mark.asyncio
async def test_async_task_isolation():
    """Each async task must have its own session_id."""
    results = {}

    async def worker(name: str, session_id: str):
        ContextManager.set_current_session_id(session_id)
        await asyncio.sleep(0.01)
        results[name] = ContextManager.get_current_session_id()

    await asyncio.gather(
        worker("A", "session-aaa"),
        worker("B", "session-bbb"),
        worker("C", "session-ccc"),
    )

    assert results["A"] == "session-aaa"
    assert results["B"] == "session-bbb"
    assert results["C"] == "session-ccc"


@pytest.mark.asyncio
async def test_context_inheritance():
    """A child task should inherit the parent's context."""
    ContextManager.set_current_session_id("main-session")

    async def inherit_worker():
        return ContextManager.get_current_session_id()

    result = await inherit_worker()
    assert result == "main-session"


@pytest.mark.asyncio
async def test_context_reset():
    """set_current_session_id(None) should reset to default."""
    ContextManager.set_current_session_id("something")
    ContextManager.set_current_session_id(None)
    assert ContextManager.get_current_session_id() is None
