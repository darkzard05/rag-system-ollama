"""Cache hits must replay the citations captured on the original answer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph._graph_core import _QUERY_CACHE_VALUE_VERSION
from core.graph.graph_builder import generate, preprocess
from core.session import SessionManager

_CITATIONS = [
    {
        "doc_id": "abc123",
        "text_span": "CM3 is a causal LM",
        "section": "모델 구조",
        "page": 3,
        "score": 0.93,
    }
]


def _config() -> dict:
    return {
        "configurable": {
            "session_id": "cache-cite",
            "thread_id": "cache-cite",
            "llm": MagicMock(),
        }
    }


def _state(**overrides: object) -> dict:
    state = {
        "input": "CM3 모델 구조를 설명해줘",
        "intent": "general",
        "route": "generate",
        "search_queries": [],
        "relevant_docs": [],
        "response": None,
        "thought": None,
        "performance": None,
        "search_weights": None,
        "is_cached": True,
        "cached_response": "캐시된 답변",
        "cached_citations": None,
        "retry_count": 0,
    }
    state.update(overrides)
    return state


def _cache_manager(value: dict) -> MagicMock:
    cm = MagicMock()
    cm.get = AsyncMock(return_value=value)
    cm.set = AsyncMock()
    cm.semantic_cache = SimpleNamespace(clear=AsyncMock())
    return cm


@pytest.mark.asyncio
async def test_generate_cache_hit_streams_stored_citations() -> None:
    """A cache hit replays citations so the UI never drops references."""
    captured: list[dict] = []
    with patch(
        "core.graph._graph_core.adispatch_custom_event",
        side_effect=lambda name, data, config=None: captured.append(
            {"name": name, "data": dict(data)}
        ),
    ):
        await generate(
            _state(cached_citations=_CITATIONS),
            _config(),
            writer=MagicMock(),
        )

    cite_events = [event for event in captured if event["name"] == "citations"]
    assert cite_events, "cache hit must emit a citations event"
    assert cite_events[0]["data"]["citations"] == _CITATIONS


@pytest.mark.asyncio
async def test_preprocess_propagates_cached_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preprocess exposes cached citations on the graph state."""
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_MIN_CONF", 0.85)
    value = {
        "response": "캐시된 답변",
        "confidence": 0.99,
        "version": _QUERY_CACHE_VALUE_VERSION,
        "citations": _CITATIONS,
    }
    with (
        patch(
            "core.graph._preprocess.get_cache_manager",
            return_value=_cache_manager(value),
        ),
        patch.object(SessionManager, "get", return_value="filehash"),
        patch(
            "core.graph._preprocess._ensure_query_cache_embedder",
            new=AsyncMock(),
        ),
    ):
        result = await preprocess(
            _state(is_cached=False, cached_response=None),
            _config(),
            writer=None,
        )

    assert result["cached_citations"] == _CITATIONS


@pytest.mark.asyncio
async def test_fresh_answer_cache_write_preserves_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache entry written after a fresh answer carries its citations."""
    monkeypatch.setattr("core.graph._generate.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph._generate.QUERY_CACHE_MIN_CONF", 0.85)
    cm = _cache_manager(None)

    with (
        patch("core.graph._generate.get_cache_manager", return_value=cm),
        patch("core.graph._generate._ensure_query_cache_embedder", new=AsyncMock()),
        patch.object(SessionManager, "get", return_value="filehash"),
    ):
        from core.graph._generate import _schedule_query_cache_set

        scheduled = _schedule_query_cache_set(
            query="CM3 모델 구조를 설명해줘",
            response="캐시된 답변",
            confidence=0.99,
            file_hash="filehash",
            citations=_CITATIONS,
        )
        await asyncio.sleep(0.01)

    assert scheduled is True
    assert cm.set.await_args is not None
    assert cm.set.await_args.args[1]["citations"] == _CITATIONS
