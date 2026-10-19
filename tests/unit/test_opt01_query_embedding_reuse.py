"""[최적화-01] 쿼리 임베딩 듀얼 라운드트립 제거 회귀 테스트.

preprocess 가 순수 쿼리 벡터를 1회 선계산해 (a) 세맨틱 캐시 조회에 주입하고
(b) state 를 통해 retrieve 가 재임베딩 없이 재사용하는지 검증한다.

기존 결함: 캐시 조회가 네임스페이스 키를, retrieve 가 순수 쿼리를 각각
임베딩해 쿼리당 Ollama HTTP 호출이 2회 직렬 발생했다.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("IS_CI_TEST", "true")

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from core.graph.graph_builder import preprocess, retrieve_and_rerank  # noqa: E402
from core.session import SessionManager  # noqa: E402

_SID = "opt01-session"


def _base_config(**configurable: Any) -> dict:
    cfg = {"session_id": _SID, "thread_id": _SID}
    cfg.update(configurable)
    return {"configurable": cfg}


def _base_state(query: str, **overrides: Any) -> dict:
    state = {
        "input": query,
        "intent": None,
        "route": "generate",
        "search_queries": [],
        "relevant_docs": [],
        "response": None,
        "thought": None,
        "performance": None,
        "search_weights": None,
        "query_embedding": None,
        "is_cached": False,
        "cached_response": None,
        "retry_count": 0,
    }
    state.update(overrides)
    return state


_PRECOMPUTED = [0.5, 0.5, 0.0, 0.0]


# ---------------------------------------------------------------------------
# (a) preprocess: 1회 선계산 후 캐시 조회에 주입
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_preprocess_injects_precomputed_embedding_into_cache(monkeypatch):
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr("core.graph._preprocess.QUERY_CACHE_MIN_CONF", 0.85)

    SessionManager.set("file_hash", "abc123", session_id=_SID)
    faiss = object()

    fake_cm = MagicMock()
    fake_cm.get = AsyncMock(return_value=None)

    embed_calls: list[tuple[Any, str]] = []

    def _fake_embed(retriever: Any, query: str) -> list[float]:
        embed_calls.append((retriever, query))
        return list(_PRECOMPUTED)

    state = _base_state("reusable query")
    try:
        with (
            patch("core.graph._preprocess.get_cache_manager", return_value=fake_cm),
            patch(
                "core.graph._preprocess._ensure_query_cache_embedder",
                new=AsyncMock(),
            ),
            patch("core.retriever_factory.embed_faiss_query_vector", new=_fake_embed),
        ):
            result = await preprocess(
                state, _base_config(faiss_retriever=faiss), writer=None
            )
    finally:
        SessionManager.delete("file_hash", session_id=_SID)

    # 순수 쿼리로 정확히 1회만 임베딩된다.
    assert embed_calls == [(faiss, "reusable query")]
    # 캐시 조회는 네임스페이스 키이되, 임베딩은 주입되어 내부 재임베딩이 없다.
    fake_cm.get.assert_called_once_with(
        "abc123:reusable query",
        use_semantic=True,
        query_embedding=_PRECOMPUTED,
    )
    # retrieve 재사용을 위한 state 채널도 채워진다.
    assert result["query_embedding"] == _PRECOMPUTED


# ---------------------------------------------------------------------------
# (b) retrieve: state 벡터 재사용 → 재임베딩 없음
# ---------------------------------------------------------------------------
def _fast_docs() -> list[Document]:
    return [
        Document(
            page_content=f"문서 {i} 본문. "
            + ("의미 있는 본문을 50자 이상으로 채운다. " * 3),
            metadata={"page": i + 1, "score": 0.9 - i * 0.01},
        )
        for i in range(15)
    ]


@pytest.mark.asyncio
async def test_retrieve_reuses_state_embedding_without_reembedding():
    captured: dict[str, Any] = {}

    async def _bm25(retriever: Any, query: str, k: int) -> list[Document]:
        return list(_fast_docs())

    async def _faiss(
        retriever: Any, query: str, k: int, query_vector: Any | None = None
    ) -> list[Document]:
        captured["query_vector"] = query_vector
        return list(_fast_docs())

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("embed_faiss_query_vector must not be called")

    reranker = AsyncMock()
    reranker.rerank.return_value = (list(_fast_docs()[:5]), None)

    state = _base_state("이 문서들의 공통 주제를 설명해줘")
    state["query_embedding"] = list(_PRECOMPUTED)
    config = _base_config(
        llm=MagicMock(),
        bm25_retriever=MagicMock(),
        faiss_retriever=MagicMock(),
    )

    with (
        patch("core.retriever_factory.search_bm25_with_scores", new=_bm25),
        patch("core.retriever_factory.search_faiss_with_scores", new=_faiss),
        patch("core.retriever_factory.embed_faiss_query_vector", new=_boom),
        patch(
            "core.async_reranker.get_async_reranker",
            new=AsyncMock(return_value=reranker),
        ),
    ):
        result = await retrieve_and_rerank(state, config, writer=None)

    assert result["relevant_docs"]
    assert captured["query_vector"] is not None
    import numpy as np

    np.testing.assert_allclose(
        np.asarray(captured["query_vector"], dtype="float32"),
        np.asarray(_PRECOMPUTED, dtype="float32"),
    )


# ---------------------------------------------------------------------------
# (c) 재작성 쿼리(재시도)에는 원본 벡터를 재사용하지 않는다
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retrieve_does_not_reuse_vector_for_rewritten_query():
    captured: dict[str, Any] = {}
    sentinel = [9.0, 8.0, 7.0, 6.0]

    async def _bm25(retriever: Any, query: str, k: int) -> list[Document]:
        return list(_fast_docs())

    async def _faiss(
        retriever: Any, query: str, k: int, query_vector: Any | None = None
    ) -> list[Document]:
        captured["query_vector"] = query_vector
        captured["query"] = query
        return list(_fast_docs())

    def _fake_embed(retriever: Any, query: str) -> list[float]:
        return list(sentinel)

    reranker = AsyncMock()
    reranker.rerank.return_value = (list(_fast_docs()[:5]), None)

    state = _base_state("원본 질의")
    state["query_embedding"] = list(_PRECOMPUTED)
    state["search_queries"] = ["재작성된 질의"]
    config = _base_config(
        llm=MagicMock(),
        bm25_retriever=MagicMock(),
        faiss_retriever=MagicMock(),
    )

    with (
        patch("core.retriever_factory.search_bm25_with_scores", new=_bm25),
        patch("core.retriever_factory.search_faiss_with_scores", new=_faiss),
        patch("core.retriever_factory.embed_faiss_query_vector", new=_fake_embed),
        patch(
            "core.async_reranker.get_async_reranker",
            new=AsyncMock(return_value=reranker),
        ),
    ):
        result = await retrieve_and_rerank(state, config, writer=None)

    assert result["relevant_docs"]
    assert captured["query"] == "재작성된 질의"
    import numpy as np

    np.testing.assert_allclose(
        np.asarray(captured["query_vector"], dtype="float32"),
        np.asarray(sentinel, dtype="float32"),
    )
