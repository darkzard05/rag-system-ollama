"""[최적화-03] FAISS 검색 이중 스레드 디스패치 단일화 회귀 테스트.

기존 결함: ``_search_faiss_pipeline`` 이 쿼리 임베딩 추출과 FAISS 검색을 각각
별도 ``asyncio.to_thread`` 로 디스패치해 연속 2회의 스레드 풀 전환 오버헤드가
발생했다. 최적화 후에는 두 작업이 ``_embed_and_search_faiss_sync`` 한 번의
스레드 호출로 묶여 실제 ``VectorStoreRetriever`` 경로에서는 ``to_thread`` 가
1회만 발생하고, ``embed_faiss_query_vector``/``_faiss_results_with_scores`` 는
개별 스레드 작업으로 디스패치되지 않아야 한다.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ.setdefault("IS_CI_TEST", "true")

import pytest  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.vectorstores.utils import DistanceStrategy  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.vectorstores import VectorStoreRetriever  # noqa: E402

from core.graph.graph_builder import retrieve_and_rerank  # noqa: E402

_TOKENS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


class _OneHotEmbedder(Embeddings):
    """테스트 전용 one-hot 임베더 (오프라인, 모델 호출 없음)."""

    def embed_query(self, text: str) -> list[float]:
        return [1.0 if t in text else 0.0 for t in _TOKENS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def _corpus() -> list[str]:
    return [
        f"only {t} unique term. "
        + ("padding sentence to exceed fifty characters. " * 2)
        for t in _TOKENS
    ]


def _real_faiss_retriever() -> VectorStoreRetriever:
    embedder = _OneHotEmbedder()
    text_embeddings = [(text, embedder.embed_query(text)) for text in _corpus()]
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedder,
        metadatas=[{"doc_id": f"fa_{t}"} for t in _TOKENS],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    return VectorStoreRetriever(
        vectorstore=vector_store,
        search_type="similarity",
        search_kwargs={"k": 3},
    )


def test_embed_and_search_faiss_sync_bundles_and_returns_timings():
    """단일 동기 헬퍼가 임베딩과 검색을 함께 수행하고 시간을 반환한다."""
    from core.graph._retrieval import _embed_and_search_faiss_sync

    retriever = _real_faiss_retriever()
    docs, embed_ms, faiss_ms = _embed_and_search_faiss_sync(retriever, "alpha", 3, None)

    assert len(docs) == 3
    assert all(isinstance(d.metadata.get("score"), float) for d in docs)
    assert docs[0].metadata["doc_id"] == "fa_alpha"
    assert embed_ms >= 0.0
    assert faiss_ms >= 0.0


def test_embed_and_search_faiss_sync_signals_fallback_for_non_vectorstore():
    """VectorStoreRetriever가 아니면 폴백 신호(None)를 반환한다."""
    from core.graph._retrieval import _embed_and_search_faiss_sync

    assert _embed_and_search_faiss_sync(AsyncMock(), "alpha", 3, None) is None


@pytest.mark.asyncio
async def test_faiss_pipeline_dispatches_single_to_thread():
    """실제 VectorStoreRetriever 경로는 to_thread를 FAISS 파이프라인당 1회만 쓴다."""
    from core.graph import _retrieval
    from core.retriever_factory import embed_faiss_query_vector

    retriever = _real_faiss_retriever()
    bm25_docs = [
        Document(page_content=text, metadata={"doc_id": f"bm_{i}"})
        for i, text in enumerate(_corpus())
    ]
    bm25 = AsyncMock()
    bm25.ainvoke.return_value = bm25_docs

    dispatched: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _spy(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        dispatched.append(func)
        return await real_to_thread(func, *args, **kwargs)

    reranker = AsyncMock()
    reranker.rerank.return_value = (bm25_docs[:3], None)

    state = {
        "input": "alpha beta 감마 델타 문서 주제를 요약해줘",
        "search_queries": [],
        "retry_count": 0,
        "query_embedding": None,
    }
    config = {
        "configurable": {
            "bm25_retriever": bm25,
            "faiss_retriever": retriever,
            "session_id": "opt03-session",
        }
    }

    with (
        patch.object(asyncio, "to_thread", _spy),
        patch(
            "core.async_reranker.get_async_reranker",
            new=AsyncMock(return_value=reranker),
        ),
    ):
        result = await retrieve_and_rerank(state, config, writer=None)

    assert result["relevant_docs"]
    # 임베딩 + 검색이 단일 to_thread 호출로 묶여 디스패치된다.
    assert dispatched.count(_retrieval._embed_and_search_faiss_sync) == 1
    # 개별 임베딩/검색 함수는 스레드 작업으로 디스패치되지 않는다.
    assert embed_faiss_query_vector not in dispatched
