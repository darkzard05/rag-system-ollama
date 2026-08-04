from dataclasses import dataclass

import pytest
from src.core.search_aggregator import (
    AggregationStrategy,
    SearchResultAggregator,
)


@dataclass
class MockResult:
    doc_id: str
    score: float
    content: str = "mock content"
    metadata: dict = None

    def __init__(self, doc_id, score, content="mock content", metadata=None):
        self.doc_id = doc_id
        self.score = score
        self.content = content
        self.metadata = metadata or {}


def test_aggregated_score_calculation():
    aggregator = SearchResultAggregator()

    search_results = {
        "faiss": [
            MockResult("doc_a", 0.9),
            MockResult("doc_b", 0.7),
        ],
        "bm25": [
            MockResult("doc_a", 20.0),
            MockResult("doc_b", 10.0),
            MockResult("doc_c", 5.0),
        ],
    }

    aggregated, metrics = aggregator.aggregate_results(
        search_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        weights={"faiss": 0.5, "bm25": 0.5},
        top_k=10,
    )

    res_map = {r.doc_id: r for r in aggregated}

    # FAISS norm: 0.9, BM25 norm: 1.0 → RRF 1/(60+1)*0.5 + 1/(60+1)*0.5 ≈ 0.0164
    assert res_map["doc_a"].aggregated_score == pytest.approx(0.0164, abs=1e-3)
    assert res_map["doc_b"].aggregated_score == pytest.approx(0.0161, abs=1e-3)
    assert res_map["doc_c"].aggregated_score == pytest.approx(0.0079, abs=1e-3)


def test_only_faiss_source():
    """Aggregation should work with a single source."""
    aggregator = SearchResultAggregator()
    search_results = {
        "faiss": [
            MockResult("doc_x", 0.8),
            MockResult("doc_y", 0.5),
        ],
    }

    aggregated, _ = aggregator.aggregate_results(
        search_results, strategy=AggregationStrategy.WEIGHTED_RRF, top_k=10
    )

    assert len(aggregated) == 2
    assert aggregated[0].doc_id == "doc_x"
