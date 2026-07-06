from dataclasses import dataclass
import pytest
from src.core.search_aggregator import (
    SearchResultAggregator,
    AggregationStrategy,
    AggregatedResult,
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


def test_confidence_score_calculation():
    aggregator = SearchResultAggregator()

    # Mock search results
    # FAISS: range [0, 1]
    # BM25: range [5, 20]
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

    # Use Weighted RRF
    aggregated, metrics = aggregator.aggregate_results(
        search_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        weights={"faiss": 0.5, "bm25": 0.5},
        top_k=10,
    )

    # Convert to dict for easy access
    res_map = {r.doc_id: r for r in aggregated}

    # Doc A:
    # FAISS: 0.9 -> norm 0.9
    # BM25: 20.0 -> norm (20-5)/(20-5) = 1.0
    # Confidence: max(0.9, 1.0) = 1.0
    assert res_map["doc_a"].confidence_score == pytest.approx(1.0)

    # Doc B:
    # FAISS: 0.7 -> norm 0.7
    # BM25: 10.0 -> norm (10-5)/(20-5) = 0.3333
    # Confidence: max(0.7, 0.3333) = 0.7
    assert res_map["doc_b"].confidence_score == pytest.approx(0.7)

    # Doc C:
    # FAISS: Not found -> 0.0
    # BM25: 5.0 -> norm (5-5)/(20-5) = 0.0
    # Confidence: 0.0
    assert res_map["doc_c"].confidence_score == pytest.approx(0.0)


def test_faiss_clipping():
    aggregator = SearchResultAggregator()
    search_results = {
        "faiss": [
            MockResult("doc_a", 1.2),  # Should be clipped to 1.0
            MockResult("doc_b", -0.2),  # Should be clipped to 0.0
        ]
    }

    aggregated, _ = aggregator.aggregate_results(
        search_results, strategy=AggregationStrategy.WEIGHTED_RRF
    )

    res_map = {r.doc_id: r for r in aggregated}
    assert res_map["doc_a"].confidence_score == 1.0
    assert res_map["doc_b"].confidence_score == 0.0


if __name__ == "__main__":
    # To run manually if pytest is not available in this env's shell directly
    import dataclasses
    from dataclasses import dataclass

    test_confidence_score_calculation()
    test_faiss_clipping()
    print("All tests passed!")
