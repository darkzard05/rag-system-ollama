"""재시도 예산 리듀서 계약 복구 검증 (R1a-01, P1-2).

리뷰 실증: grade 예외 경로(LLM 파싱 실패 1회)가 `rewrite_query` 폴백과 합쳐져
retry_count가 단일 사이클에 0→3으로 이중 소진됐다. 근본 원인은 폴백이 합산
리듀서(reset_or_add)에 "절대 목표값" `min(retry_count + 1, 3)`을 델타로 반환한 것.

- (a) grade NO → rewrite 폴백 경로에서 retry_count가 0→1(단일 증가)만 되어야 한다.
- (b) `max_retries: 2` 하드캡 도달 시 3번째 재작성이 발생하지 않아야 한다.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from api.schemas import reset_or_add
from core.graph_builder import grade_documents, rewrite_query


@pytest.fixture
def failing_llm() -> tuple[MagicMock, AsyncMock]:
    """grade의 LLM JSON 파싱이 항상 실패하도록 모킹 (예외 경로 재현)."""
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    json_llm.ainvoke.side_effect = ValueError("JSON 파싱 실패")
    llm.ainvoke.return_value = SimpleNamespace(content="JSON 없는 응답")
    return llm, json_llm


def _grade_state(**overrides: object) -> dict[str, object]:
    """grade_documents 단위 호출용 상태 (rerank_score < min_score_to_skip)."""
    state: dict[str, object] = {
        "input": "테스트 질문",
        "relevant_docs": [
            Document(page_content="무관한 내용", metadata={"rerank_score": 0.1}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_rewrite_fallback_single_increment(
    failing_llm: tuple[MagicMock, AsyncMock],
) -> None:
    """(a) grade 예외(NO) → rewrite 폴백 경로: retry_count는 0→1 단일 증가여야 한다.

    현재 결함: rewrite 폴백이 `min(retry_count + 1, 3) = 2`를 델타로 반환해
    `reset_or_add(1, 2) = 3`으로 단일 사이클 0→3 소진. 폴백은 `{}`만 반환해야 한다.
    """
    llm, _json_llm = failing_llm
    config = {"configurable": {"llm": llm}}

    # grade: LLM 파싱 실패 → transform + retry_count 델타 +1
    grade_result = await grade_documents(_grade_state(), config, writer=None)
    assert grade_result["intent"] == "transform"
    assert grade_result["retry_count"] == 1

    # rewrite 폴백: grade가 search_queries를 생성하지 못한 경로
    state: dict[str, object] = {
        "input": "테스트 질문",
        "search_queries": [],
        "retry_count": reset_or_add(0, grade_result["retry_count"]),
    }
    rewrite_result = await rewrite_query(state, config, writer=None)
    assert "retry_count" not in rewrite_result  # 폴백은 증가 델타를 반환하지 않는다

    # rewrite가 retry_count를 반환하지 않으면 리듀서가 호출되지 않으므로 상태가 그대로 유지된다.
    final_count = state["retry_count"]
    if "retry_count" in rewrite_result:
        final_count = reset_or_add(final_count, rewrite_result["retry_count"])
    assert final_count == 1  # 결함 시 3 (0→3 단일 사이클 소진)


@pytest.mark.asyncio
async def test_hardcap_prevents_third_retry() -> None:
    """(b) max_retries=2 하드캡: 3번째 재작성이 발생하지 않고 retry_count=2에서 종료.

    LLM이 계속 실패해도 grade는 `retry_count >= max_retries`에서 즉시 generate로
    전환해야 한다. 재작성 횟수는 정확히 2회(max_retries), retry_count는 2가 되어야
    한다. 현재 결함 시 rewrite 1회 + retry_count=3으로 실패.
    """
    from core.graph_builder import (
        build_graph,
        invalidate_graph_cache,
    )
    from core.graph_builder import (
        rewrite_query as _rewrite_query,
    )

    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    json_llm.ainvoke.side_effect = ValueError("JSON 파싱 실패")
    llm.ainvoke.return_value = SimpleNamespace(content="JSON 없는 응답")

    async def mock_astream(*_args: object, **_kwargs: object):
        chunk = MagicMock()
        chunk.content = "재시도 상한 도달 후 생성된 답변"
        chunk.response_metadata = {"prompt_eval_count": 10}
        yield chunk

    llm.astream = mock_astream

    def mock_convert(chunk: Any) -> tuple[Any, None]:
        return chunk.content, None

    llm._convert_chunk_to_thought_and_content = mock_convert

    bm25 = AsyncMock()
    faiss = AsyncMock()
    bm25.ainvoke.return_value = [
        Document(page_content="무관한 내용", metadata={"rerank_score": 0.1})
    ]
    faiss.ainvoke.return_value = []

    rewrite_calls: list[int] = []

    async def counting_rewrite(state, config, *, writer=None):
        rewrite_calls.append(1)
        return await _rewrite_query(state, config, writer=writer)

    invalidate_graph_cache()
    try:
        with (
            patch(
                "core.async_reranker.get_async_reranker", new_callable=AsyncMock
            ) as mock_get,
            patch("core.graph_builder.rewrite_query", new=counting_rewrite),
            patch("aiosqlite.connect", side_effect=Exception("force InMemorySaver")),
        ):
            reranker = AsyncMock()
            reranker.rerank.side_effect = lambda docs, **kwargs: (docs, None)
            mock_get.return_value = reranker

            graph = await build_graph()
            config = {
                "configurable": {
                    "llm": llm,
                    "bm25_retriever": bm25,
                    "faiss_retriever": faiss,
                    "thread_id": "test_hardcap",
                }
            }
            result = await graph.ainvoke(
                {"input": "질문", "chat_history": []}, config=config
            )
    finally:
        invalidate_graph_cache()

    assert result["retry_count"] == 2  # 결함 시 3
    assert len(rewrite_calls) == 2  # max_retries=2 → 3번째 재작성 없음
    assert "재시도 상한 도달 후 생성된 답변" in result["response"]
