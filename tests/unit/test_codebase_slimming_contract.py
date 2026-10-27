import pytest
from dataclasses import dataclass, field
from ui.components.chat_autoscroll import (
    edge_latch,
    flag_read,
    stick_math,
    BOTTOM_STICK_PX,
)
from core.search_aggregator import (
    SearchResultAggregator,
    AggregationStrategy,
    AggregatedResult,
)


@dataclass
class DummySearchHit:
    doc_id: str
    content: str
    score: float
    node_id: str = "test_node"
    metadata: dict = field(default_factory=dict)


def test_autoscroll_helpers_contract():
    # stick_math 계약 보존 검증
    res = stick_math(scroll_height=1000, scroll_top=995, client_height=0)
    assert res == {"dist": -1, "stick_active": False}

    res = stick_math(
        scroll_height=1000, scroll_top=995, client_height=5, threshold=BOTTOM_STICK_PX
    )
    assert res == {"dist": 0, "stick_active": True}

    # edge_latch 계약 보존 검증
    assert edge_latch(True, 0, BOTTOM_STICK_PX + 1) is True
    assert edge_latch(True, 0, BOTTOM_STICK_PX) is False

    # flag_read 계약 보존 검증
    assert flag_read(True, "msg-123", "msg-123") is True
    assert flag_read(True, "msg-123", "msg-456") is False


def test_rrf_fusion_single_path_equivalence():
    aggregator = SearchResultAggregator()
    doc1_bm25 = DummySearchHit(doc_id="doc1", content="c1", score=0.9)
    doc2_bm25 = DummySearchHit(doc_id="doc2", content="c2", score=0.8)
    doc2_faiss = DummySearchHit(doc_id="doc2", content="c2", score=0.9)
    doc1_faiss = DummySearchHit(doc_id="doc1", content="c1", score=0.8)

    search_results = {"bm25": [doc1_bm25, doc2_bm25], "faiss": [doc2_faiss, doc1_faiss]}

    results, metrics = aggregator.aggregate_results(
        search_results=search_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=2,
        weights={"bm25": 0.4, "faiss": 0.6},
    )

    assert len(results) == 2
    assert metrics.nodes_processed == 2
    expected_doc1 = 0.4 / 61.0 + 0.6 / 62.0
    expected_doc2 = 0.4 / 62.0 + 0.6 / 61.0
    scores = {r.doc_id: r.aggregated_score for r in results}
    assert pytest.approx(scores["doc1"], 1e-6) == expected_doc1
    assert pytest.approx(scores["doc2"], 1e-6) == expected_doc2
