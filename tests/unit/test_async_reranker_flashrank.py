"""FlashRank 크로스-인코더 리랭커 단위 테스트.

- (a) 점수 매핑·top_k 정렬·(documents, scores) 2-tuple 반환
- (b) FlashRank 예외 시 AsyncSemanticReranker 폴백 (engine=auto)
- (c) engine="semantic"이면 bi-encoder 경로 유지
- (d) graph_builder.retrieve_and_rerank가 cross-encoder 경로 사용 (get_async_reranker patch)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

import core.async_reranker as ar
from core.async_reranker import (
    AsyncCrossEncoderReranker,
    AsyncSemanticReranker,
    get_async_reranker,
)


@pytest.fixture(autouse=True)
def _reset_global_reranker():
    """모듈 전역 _async_reranker/_rerank_engine_active/_semantic_fallback_reranker를 초기화합니다."""
    ar._async_reranker = None
    ar._rerank_engine_active = "flashrank"
    ar._semantic_fallback_reranker = None
    yield
    ar._async_reranker = None
    ar._rerank_engine_active = "flashrank"
    ar._semantic_fallback_reranker = None


def _make_docs(n: int = 3) -> list[Document]:
    return [
        Document(page_content=f"문서 내용 {i}", metadata={"page": i + 1})
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_cross_encoder_rerank_maps_scores_and_sorts_topk():
    """(a) FlashRank 점수 매핑, top_k 정렬, (documents, scores) 2-tuple 반환"""
    docs = _make_docs(3)
    fake_results = [
        {"id": 1, "text": docs[1].page_content, "score": 0.6},
        {"id": 0, "text": docs[0].page_content, "score": 0.4},
        {"id": 2, "text": docs[2].page_content, "score": 0.9},
    ]
    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = fake_results

    reranker = AsyncCrossEncoderReranker()
    with patch(
        "core.model_loader.ModelManager.get_flashranker", new_callable=AsyncMock
    ) as mock_get_flashranker:
        mock_get_flashranker.return_value = mock_ranker

        result = await reranker.rerank(docs, query="테스트 질문", top_k=2)

    assert isinstance(result, tuple)
    assert len(result) == 2
    ranked_docs, ranked_scores = result
    assert isinstance(ranked_docs, list)
    assert isinstance(ranked_scores, list)
    assert all(isinstance(s, float) for s in ranked_scores)
    assert [d.page_content for d in ranked_docs] == [
        docs[2].page_content,
        docs[1].page_content,
    ]
    assert ranked_scores == [0.9, 0.6]
    assert ranked_docs[0].metadata["rerank_score"] == 0.9
    assert ranked_docs[1].metadata["rerank_score"] == 0.6


@pytest.mark.asyncio
async def test_cross_encoder_falls_back_to_semantic_on_flashrank_error():
    """(b) FlashRank 로드 실패 시 AsyncSemanticReranker로 폴백"""
    docs = _make_docs(3)
    reranker = AsyncCrossEncoderReranker()

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [1.0, 0.0]
    mock_embedder.embed_documents.return_value = [[1.0, 0.0]] * len(docs)

    with (
        patch(
            "core.model_loader.ModelManager.get_flashranker",
            new_callable=AsyncMock,
            side_effect=RuntimeError("ONNX 모델 로드 실패"),
        ),
        patch(
            "core.model_loader.ModelManager.get_embedder", new_callable=AsyncMock
        ) as mock_get_embedder,
    ):
        mock_get_embedder.return_value = mock_embedder

        ranked_docs, ranked_scores = await reranker.rerank(
            docs, query="테스트 질문", top_k=2
        )

    assert len(ranked_docs) == 2
    assert isinstance(ranked_scores, list)
    assert all(isinstance(s, float) for s in ranked_scores)
    mock_embedder.embed_query.assert_called_once()
    assert mock_embedder.embed_documents.called


@pytest.mark.asyncio
async def test_get_async_reranker_semantic_engine_returns_semantic():
    """(c) engine="semantic"이면 AsyncSemanticReranker(bi-encoder) 반환"""
    with (
        patch("core.async_reranker.RERANKER_ENGINE", "semantic"),
        patch(
            "core.model_loader.ModelManager.get_embedder", new_callable=AsyncMock
        ) as mock_get_embedder,
    ):
        mock_get_embedder.return_value = MagicMock()

        reranker = await get_async_reranker()

    assert isinstance(reranker, AsyncSemanticReranker)
    assert not isinstance(reranker, AsyncCrossEncoderReranker)


@pytest.mark.asyncio
async def test_retrieve_and_rerank_uses_cross_encoder_in_auto_mode():
    """(d) retrieve_and_rerank가 get_async_reranker 반환 랭커의 rerank를 사용"""
    from core.graph_builder import retrieve_and_rerank

    docs = _make_docs(5)
    bm25 = AsyncMock()
    bm25.ainvoke.return_value = docs
    faiss = AsyncMock()
    faiss.ainvoke.return_value = []

    fake_results = [
        {"id": i, "text": doc.page_content, "score": 0.9 - i * 0.1}
        for i, doc in enumerate(docs)
    ]
    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = fake_results

    config = {
        "configurable": {
            "bm25_retriever": bm25,
            "faiss_retriever": faiss,
            "session_id": "test-cross-encoder",
        }
    }
    state = {
        "input": "크로스 인코더 리랭킹 경로 검증 질문입니다",
        "search_queries": [],
        "retry_count": 0,
    }

    cross_encoder = AsyncCrossEncoderReranker()
    with (
        patch(
            "core.async_reranker.get_async_reranker",
            new_callable=AsyncMock,
            return_value=cross_encoder,
        ),
        patch(
            "core.model_loader.ModelManager.get_flashranker",
            new_callable=AsyncMock,
            return_value=mock_ranker,
        ),
    ):
        result = await retrieve_and_rerank(state, config, writer=None)

    assert result["relevant_docs"]
    assert mock_ranker.rerank.called
