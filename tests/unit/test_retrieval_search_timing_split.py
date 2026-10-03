"""검색 단계 소스별 소요시간 분할 계측 회귀 테스트.

``search_ms`` 단일 지표로는 2026-10-01 관측된 비결정적 스파이크(동일 쿼리
``cm3``가 80.7 / 817.8 / 93.3ms)의 원인이 임베딩 라운드트립인지 FAISS 연산인지
구분할 수 없었다. ``bm25_ms`` / ``faiss_ms``로 쪼개 두었으므로, 느린 소스의 시간이
정확히 해당 키에 격리되고 다른 소스 키는 작게 유지되는지를 고정한다.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("IS_CI_TEST", "true")

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from core.graph.graph_builder import retrieve_and_rerank  # noqa: E402

_FAST_DOCS = [
    Document(
        page_content=f"문서 {i} 본문. "
        + ("의미 있는 본문을 50자 이상으로 채운다. " * 3),
        metadata={"page": i + 1, "score": 0.9 - i * 0.01},
    )
    for i in range(15)
]


@pytest.mark.asyncio
async def test_search_timing_attributes_slow_source_to_its_own_key():
    """느린 FAISS 소스의 시간이 faiss_ms에만 실리고 bm25_ms는 빠르게 유지된다."""
    captured: list[str] = []

    async def _fast_bm25(retriever: Any, query: str, k: int) -> list[Document]:
        return list(_FAST_DOCS)

    async def _slow_faiss(retriever: Any, query: str, k: int) -> list[Document]:
        await asyncio.sleep(0.25)
        return list(_FAST_DOCS)

    state = {
        "input": "이 문서들의 공통 주제를 설명해줘",
        "search_queries": [],
        "retry_count": 0,
    }
    config = {
        "configurable": {
            "llm": MagicMock(),
            "bm25_retriever": MagicMock(),
            "faiss_retriever": MagicMock(),
        }
    }

    reranker = AsyncMock()
    reranker.rerank.return_value = (list(_FAST_DOCS[:5]), None)

    with (
        patch("core.retriever_factory.search_bm25_with_scores", new=_fast_bm25),
        patch("core.retriever_factory.search_faiss_with_scores", new=_slow_faiss),
        patch(
            "core.async_reranker.get_async_reranker",
            new=AsyncMock(return_value=reranker),
        ),
        patch("core.graph._retrieval.logger") as mock_logger,
    ):
        mock_logger.debug.side_effect = captured.append
        mock_logger.info.side_effect = captured.append
        result = await retrieve_and_rerank(state, config, writer=None)

    assert result["relevant_docs"]

    timing_line = next(
        (line for line in captured if "[RETRIEVE][TIMING]" in str(line)), None
    )
    assert timing_line is not None, captured

    matched = re.search(r"bm25_ms=([\d.]+)\s+faiss_ms=([\d.]+)", str(timing_line))
    assert matched is not None, timing_line
    bm25_ms = float(matched.group(1))
    faiss_ms = float(matched.group(2))

    assert faiss_ms >= 200.0, (
        f"느린 FAISS 시간이 faiss_ms에 반영되지 않음: {timing_line}"
    )
    assert bm25_ms < 100.0, f"bm25_ms가 느린 소스에 오염됨: {timing_line}"
    assert faiss_ms > bm25_ms, timing_line
    assert "embed_ms=" in str(timing_line), timing_line


@pytest.mark.asyncio
async def test_search_timing_reports_zero_for_unconfigured_source():
    """미구성 소스의 키는 n/a로 표기된다 (키 누락 방지)."""
    captured: list[str] = []

    async def _fast_bm25(retriever: Any, query: str, k: int) -> list[Document]:
        return list(_FAST_DOCS)

    state = {
        "input": "bm25 단일 소스로만 검색하는 경우",
        "search_queries": [],
        "retry_count": 0,
    }
    config = {
        "configurable": {
            "llm": MagicMock(),
            "bm25_retriever": MagicMock(),
            "faiss_retriever": None,
        }
    }

    reranker = AsyncMock()
    reranker.rerank.return_value = (list(_FAST_DOCS[:5]), None)

    with (
        patch("core.retriever_factory.search_bm25_with_scores", new=_fast_bm25),
        patch(
            "core.async_reranker.get_async_reranker",
            new=AsyncMock(return_value=reranker),
        ),
        patch("core.graph._retrieval.logger") as mock_logger,
    ):
        mock_logger.debug.side_effect = captured.append
        mock_logger.info.side_effect = captured.append
        result = await retrieve_and_rerank(state, config, writer=None)

    assert result["relevant_docs"]

    timing_line = next(
        (line for line in captured if "[RETRIEVE][TIMING]" in str(line)), None
    )
    assert timing_line is not None, captured
    assert "bm25_ms=" in str(timing_line), timing_line
    assert "faiss_ms=n/a" in str(timing_line), timing_line
    assert "embed_ms=n/a" in str(timing_line), timing_line
