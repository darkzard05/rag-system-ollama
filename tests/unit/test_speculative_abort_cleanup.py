"""An aborted stream must not leave a speculative answer for the next turn."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph._speculative_gen import _spec_registry, _SpecGenerate
from core.rag_core import RAGSystem


async def _old_answer() -> dict[str, str]:
    return {"response": "이전 질문의 답변"}


async def _aborting_stream(*args: object, **kwargs: object):  # noqa: ARG001
    raise asyncio.CancelledError
    yield  # pragma: no cover - makes this an async generator


@pytest.mark.asyncio
async def test_aborted_astream_reaps_speculative_entry() -> None:
    """A client disconnect drops the warm generate bound to that session."""
    sid = "spec_abort_cleanup_session"
    task = asyncio.create_task(_old_answer())
    await task
    _spec_registry[sid] = _SpecGenerate(
        task=task, buffer=[], created_at=time.monotonic()
    )

    with (
        patch("core.rag_core.prepare_query_config_or_build", new_callable=AsyncMock),
        patch("core.session.SessionManager.get", return_value=None),
        patch.object(RAGSystem, "_get_rag_engine", new_callable=AsyncMock) as mock_eng,
        patch("core.rag_core._stream_with_retry", new=_aborting_stream),
    ):
        mock_eng.return_value = MagicMock()
        rag = RAGSystem(session_id=sid)
        gen = await rag.astream("첫 번째 질문")
        with pytest.raises(asyncio.CancelledError):
            async for _ in gen:
                pass

    assert sid not in _spec_registry


@pytest.mark.asyncio
async def test_aborted_aquery_reaps_speculative_entry() -> None:
    """A cancelled non-streaming query must drop its warm generate too."""
    sid = "spec_abort_cleanup_aquery_session"
    task = asyncio.create_task(_old_answer())
    await task
    _spec_registry[sid] = _SpecGenerate(
        task=task, buffer=[], created_at=time.monotonic()
    )

    engine = MagicMock()
    engine.ainvoke = AsyncMock(side_effect=asyncio.CancelledError)
    with (
        patch("core.rag_core.prepare_query_config_or_build", new_callable=AsyncMock),
        patch("core.session.SessionManager.get", return_value=None),
        patch.object(RAGSystem, "_get_rag_engine", new_callable=AsyncMock) as mock_eng,
    ):
        mock_eng.return_value = engine
        rag = RAGSystem(session_id=sid)
        with pytest.raises(asyncio.CancelledError):
            await rag.aquery("첫 번째 질문")

    assert sid not in _spec_registry

    from core.graph._generate import generate as real_generate

    cached_state = {
        "input": "두 번째 질문",
        "is_cached": True,
        "cached_response": "올바른 캐시 답변",
        "cached_citations": [],
        "relevant_docs": [],
        "intent": "general",
    }
    config = {"configurable": {"thread_id": sid, "session_id": sid, "llm": MagicMock()}}
    with patch("core.graph._generate._dispatch_event", new_callable=AsyncMock):
        result = await real_generate(cached_state, config, writer=MagicMock())
    assert result["response"] == "올바른 캐시 답변"
