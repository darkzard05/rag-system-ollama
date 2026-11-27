"""A memoized transform decision must replay its optimized query."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.graph.graph_builder import grade_documents
from core.session import SessionManager

_OPTIMIZED = "CM3 인필링 프롬프트 구조"


def _state() -> dict:
    return {
        "input": "CM3 이미지 인필링",
        "relevant_docs": [
            Document(page_content="관련 없는 본문", metadata={"rerank_score": 0.3}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
        "search_queries": [],
    }


def _llm() -> MagicMock:
    llm = MagicMock()
    bound = AsyncMock()
    bound.ainvoke.return_value = SimpleNamespace(
        content=json.dumps(
            {
                "action": "rewrite",
                "is_relevant": False,
                "relevant_entities": [],
                "reason": "문서가 질문과 다름",
                "optimized_query": _OPTIMIZED,
            }
        )
    )
    llm.bind.return_value = bound
    return llm, bound


@pytest.mark.asyncio
async def test_grade_memo_hit_replays_optimized_query() -> None:
    """The second grade reuses the rewrite query instead of looping on the original."""
    session_id = "grade-memo-transform-replay"
    SessionManager.delete_session(session_id)
    llm, bound = _llm()
    config = {
        "configurable": {
            "llm": llm,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }

    first = await grade_documents(_state(), config, writer=None)
    assert first["route"] == "transform"
    assert first["search_queries"] == [_OPTIMIZED]

    second = await grade_documents(_state(), config, writer=None)

    assert second["route"] == "transform"
    assert second.get("search_queries") == [_OPTIMIZED]
    bound.ainvoke.assert_awaited_once(), "memo hit must not call the grade LLM again"

    SessionManager.delete_session(session_id)


@pytest.mark.asyncio
async def test_grade_error_path_memoizes_optimized_query() -> None:
    """A grading failure must not burn the retry budget on the same query."""
    session_id = "grade-memo-error-replay"
    SessionManager.delete_session(session_id)
    config = {
        "configurable": {
            "llm": MagicMock(),
            "session_id": session_id,
            "thread_id": session_id,
        }
    }
    failing_invoke = AsyncMock(side_effect=RuntimeError("LLM is not initialized"))
    try:
        with patch("core.graph._grade_verify._safe_invoke", new=failing_invoke):
            first = await grade_documents(_state(), config, writer=None)
            second = await grade_documents(_state(), config, writer=None)

        assert first["route"] == "transform"
        assert first.get("search_queries") == [_state()["input"]]
        assert second["route"] == "transform"
        assert second.get("search_queries") == [_state()["input"]]
        failing_invoke.assert_awaited_once(), "memo hit must not retry the failing LLM"
    finally:
        SessionManager.delete_session(session_id)
