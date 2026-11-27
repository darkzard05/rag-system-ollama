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
async def test_grade_error_path_does_not_poison_memo() -> None:
    """A grading failure must retry the LLM instead of replaying the input query."""
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
        assert first.get("search_queries") == []
        assert second["route"] == "transform"
        assert second.get("search_queries") == []
        assert failing_invoke.await_count == 2, "실패한 판단을 메모로 재생하면 안 된다"
    finally:
        SessionManager.delete_session(session_id)


@pytest.mark.asyncio
async def test_grade_without_distinct_rewrite_does_not_poison_memo() -> None:
    """최적화 쿼리가 없으면 원본 질의를 메모에 저장해 루프를 만들지 않는다."""
    session_id = "grade-memo-no-rewrite-replay"
    SessionManager.delete_session(session_id)
    llm, bound = _llm()
    bound.ainvoke.return_value = SimpleNamespace(
        content=json.dumps(
            {
                "action": "rewrite",
                "is_relevant": False,
                "relevant_entities": [],
                "reason": "문서가 질문과 다름",
                "optimized_query": None,
            }
        )
    )
    config = {
        "configurable": {
            "llm": llm,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }
    try:
        first = await grade_documents(_state(), config, writer=None)
        second = await grade_documents(_state(), config, writer=None)
        assert first["route"] == "transform"
        assert second["route"] == "transform"
        assert bound.ainvoke.await_count == 2, "원본 질의 메모 히트는 LLM을 건너뛴다"
    finally:
        SessionManager.delete_session(session_id)
