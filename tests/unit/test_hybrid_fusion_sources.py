"""하이브리드 RRF 2-노드 퓨전 소스별 점수 복원 검증 (리뷰 R3a-01 / R3a-04).

버그 배경: ``retrieve_and_rerank``가 ``aggregate_results``에 ``{"all": all_docs}``
단일 노드 dict를 전달해 ``_rrf_fusion_2node``가 활성화되지 않았고, 리트리버가
점수를 메타데이터에 주입하지 않는 경로로 모든 문서가 ``score=0.5`` 폴백에 묻혀
BM25가 top-25를 독점하고 FAISS(의미론) 결과가 실질 배제됐다.

이 파일은 세 부분으로 구성된다.
1. ``test_two_node_rrf_contract_*``  — 집계기 계약 특성화(baseline): 두 소스 dict가
   RRF 2-노드 fast-path에서 가중치·소스별 순위를 실제 반영함을 고정한다.
   (수정 전에도 통과 — 설계 계약의 기준선 문서화)
2. ``test_retrieve_and_rerank_preserves_faiss_in_top_candidates`` — 회귀(Integration):
   프로덕션 호출부가 소스별 dict + 실제 점수를 전달해 FAISS 상위 문서가 리랭킹
   후보군에서 배제되지 않는지 검증한다. (수정 전 실패 → 수정 후 통과)
3. ``test_*_score_injection_*`` — 실제 리트리버(또는 동일 계약의 FAISS) 경유 시
   점수가 ``metadata["score"]``로 주입되는지 검증한다. (수정 전 ImportError/실패)
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from core.graph_builder import retrieve_and_rerank
from core.search_aggregator import AggregationStrategy, SearchResultAggregator

# NOTE: search_bm25_with_scores / search_faiss_with_scores는 점수 주입 테스트 내부에서
# 지연 import한다 (수정 전에는 존재하지 않아 해당 테스트만 Red로 만든다).


@dataclass
class MockResult:
    """소스 노드별 검색 결과를 모사하는 최소 객체 (node_id 포함)."""

    doc_id: str
    score: float
    node_id: str = "mock"
    content: str = "mock content"
    metadata: dict | None = None

    def __init__(
        self,
        doc_id: str,
        score: float,
        node_id: str = "mock",
        content: str = "mock content",
        metadata: dict | None = None,
    ):
        self.doc_id = doc_id
        self.score = score
        self.node_id = node_id
        self.content = content
        self.metadata = metadata or {}


def _scored_docs(
    prefix: str, count: int, top_score: float, decay: float
) -> list[Document]:
    """점수 하락 순으로 정렬된 문서 목록을 생성합니다.

    실제 청크 문서처럼 page/chunk_index/start_index/end_index 메타데이터를
    포함해 ``_merge_adjacent_chunks``가 페이지 경계를 인식하도록 한다.
    """
    return [
        Document(
            page_content=f"{prefix} 문서 #{i}",
            metadata={
                "doc_id": f"{prefix}_{i}",
                "score": round(top_score - i * decay, 4),
                "source": f"{prefix}.pdf",
                "page": i,
                "chunk_index": i,
                "start_index": i * 100,
                "end_index": i * 100 + 99,
            },
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# 1. 집계기 계약 특성화(baseline) — 두 소스 dict → _rrf_fusion_2node 계약 고정
# ---------------------------------------------------------------------------


def test_two_node_rrf_contract_applies_weights_and_source_ranks():
    """가중치와 소스별 순위가 RRF 산식 weight/(k+rank)로 실제 반영된다."""
    aggregator = SearchResultAggregator()
    search_results = {
        "bm25": [
            MockResult("bm_0", 40.0, node_id="bm25"),
            MockResult("bm_1", 30.0, node_id="bm25"),
        ],
        "faiss": [
            MockResult("fa_0", 0.95, node_id="faiss"),
            MockResult("fa_1", 0.90, node_id="faiss"),
            MockResult("fa_2", 0.85, node_id="faiss"),
        ],
    }

    aggregated, metrics = aggregator.aggregate_results(
        search_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=5,
        weights={"bm25": 0.4, "faiss": 0.6},
    )

    res_map = {r.doc_id: r for r in aggregated}

    # 가중치·소스별 순위가 RRF 산식에 실제 반영
    assert res_map["fa_0"].aggregated_score == pytest.approx(0.6 / 61, abs=1e-4)
    assert res_map["fa_1"].aggregated_score == pytest.approx(0.6 / 62, abs=1e-4)
    assert res_map["bm_0"].aggregated_score == pytest.approx(0.4 / 61, abs=1e-4)

    # 같은 1순위라도 FAISS 가중치(0.6) > BM25(0.4) → fa_0가 최상위
    assert aggregated[0].doc_id == "fa_0"
    # 소스 내 순위가 유지됨: fa_0 > fa_1 > fa_2
    assert [r.doc_id for r in aggregated[:3]] == ["fa_0", "fa_1", "fa_2"]
    # 두 소스의 상위 문서가 모두 후보에 포함
    assert {r.doc_id for r in aggregated} == {"fa_0", "fa_1", "fa_2", "bm_0", "bm_1"}
    assert metrics.nodes_processed == 2


def test_two_node_rrf_contract_tracks_source_ranks():
    """동일 소스 순위가 동일 가중치에서 동일 RRF 점수를 산출한다(계약 고정)."""
    aggregator = SearchResultAggregator()
    search_results = {
        "bm25": [MockResult("bm_0", 50.0, node_id="bm25")],
        "faiss": [MockResult("fa_0", 0.9, node_id="faiss")],
    }

    aggregated, _ = aggregator.aggregate_results(
        search_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=2,
        weights={"bm25": 0.5, "faiss": 0.5},
    )

    res_map = {r.doc_id: r for r in aggregated}
    assert res_map["bm_0"].aggregated_score == pytest.approx(0.5 / 61, abs=1e-4)
    assert res_map["fa_0"].aggregated_score == pytest.approx(0.5 / 61, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. 회귀(Integration) — retrieve_and_rerank 호출부 계약 복원 검증 (Red → Green)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_and_rerank_preserves_faiss_in_top_candidates():
    """소스별 점수가 실제 반영되어 FAISS 상위 문서가 리랭킹 후보에서 배제되지 않는다.

    BM25 25건 + FAISS 8건, 동일 가중치(0.5/0.5). 수정 전(단일 노드 병합 + score=0.5
    폴백)에는 BM25가 top-25를 채워 FAISS가 0건으로 실질 배제된다. 수정 후에는
    2-노드 RRF 퓨전으로 두 소스의 상위 문서가 후보군에 공존한다.
    """
    bm25 = AsyncMock()
    faiss = AsyncMock()
    bm25.ainvoke.return_value = _scored_docs("bm", 25, top_score=25.0, decay=1.0)
    faiss.ainvoke.return_value = _scored_docs("fa", 8, top_score=1.0, decay=0.01)

    state = {
        "input": "혼합 검색",
        "search_queries": [],
        "search_weights": {"bm25": 0.5, "faiss": 0.5},
        "retry_count": 0,
    }
    config = {"configurable": {"bm25_retriever": bm25, "faiss_retriever": faiss}}

    with patch(
        "core.async_reranker.get_async_reranker", new_callable=AsyncMock
    ) as mock_get:
        reranker = AsyncMock()
        reranker.rerank.side_effect = lambda docs, **kwargs: (docs[:5], None)
        mock_get.return_value = reranker
        result = await retrieve_and_rerank(state, config, writer=None)

    relevant = result["relevant_docs"]
    assert relevant, "후보 문서가 비어서는 안 된다"
    doc_ids = [d.metadata.get("doc_id") for d in relevant]

    # FAISS 상위 문서가 후보로 살아남아야 한다 (수정 전: 0건 → 실패)
    assert any(did.startswith("fa_") for did in doc_ids), doc_ids
    # BM25 상위 문서도 공존 (두 소스 퓨전)
    assert any(did.startswith("bm_") for did in doc_ids), doc_ids
    # 실제 점수가 전달되어 리랭킹 후보에 반영된다
    assert all(isinstance(d.metadata.get("score"), float) for d in relevant)


# ---------------------------------------------------------------------------
# 3. 점수 주입 — 실제 리트리버 내부에서 스코어 캡처 (Red → Green)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bm25_score_injection_uses_real_rank_bm25_scores():
    """실제 BM25Retriever 경유 시 rank_bm25 점수가 metadata['score']로 주입된다."""
    from langchain_community.retrievers import BM25Retriever

    from core.retriever_factory import search_bm25_with_scores

    docs = [
        Document(
            page_content=f"alpha attention mechanism document number {i}",
            metadata={"doc_id": f"bm_real_{i}"},
        )
        for i in range(6)
    ]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 5

    scored = await search_bm25_with_scores(retriever, "alpha attention", k=5)

    assert len(scored) == 5
    assert all(isinstance(d.metadata.get("score"), float) for d in scored)
    scores = [d.metadata["score"] for d in scored]
    # rank_bm25 점수 내림차순이 유지된다
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_faiss_score_injection_unwraps_vectorstore():
    """FAISS VectorStoreRetriever 경유 시 유사도 점수가 metadata['score']로 주입된다."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStoreRetriever

    from core.retriever_factory import search_faiss_with_scores

    tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

    class _OneHotEmbedder(Embeddings):
        """테스트 전용 one-hot 임베더 (오프라인, 모델 호출 없음)."""

        def embed_query(self, text: str) -> list[float]:
            return [1.0 if t in text else 0.0 for t in tokens]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    embedder = _OneHotEmbedder()
    texts = [f"only {t} unique term" for t in tokens]
    text_embeddings = [(text, embedder.embed_query(text)) for text in texts]
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedder,
        metadatas=[{"doc_id": f"fa_real_{t}"} for t in tokens],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    retriever = VectorStoreRetriever(
        vectorstore=vector_store,
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    scored = await search_faiss_with_scores(retriever, "alpha", k=3)

    assert len(scored) == 3
    assert all(isinstance(d.metadata.get("score"), float) for d in scored)
    # MAX_INNER_PRODUCT: 최상위 문서가 쿼리와 가장 유사한 문서여야 한다
    assert scored[0].metadata["doc_id"] == "fa_real_alpha"
    scores = [d.metadata["score"] for d in scored]
    assert scores == sorted(scores, reverse=True)
