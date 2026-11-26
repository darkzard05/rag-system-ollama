"""Regression: one retrieval node must contribute at most one RRF rank per doc.

Two chunks with the same stable doc_id (for example repeated headers or
footers on different pages) used to be scored twice by the same retriever. The
duplicate inflated the document above genuinely distinct results and the
second chunk's content was dropped from the aggregated result.
"""

from dataclasses import dataclass

import pytest

from core.search_aggregator import AggregationStrategy, SearchResultAggregator


@dataclass
class _Result:
    doc_id: str
    score: float
    content: str = ""
    node_id: str = "mock"
    metadata: dict | None = None


def test_duplicate_doc_id_gets_one_rank_per_node() -> None:
    results, _ = SearchResultAggregator().aggregate_results(
        {
            "bm25": [_Result("same", 1.0), _Result("same", 0.9)],
            "faiss": [_Result("other", 0.7)],
        },
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=10,
        weights={"bm25": 0.5, "faiss": 0.5},
    )
    by_id = {result.doc_id: result for result in results}

    assert set(by_id) == {"same", "other"}
    assert by_id["same"].occurrence_count == 1
    assert by_id["same"].original_scores == [1.0]
    assert by_id["same"].aggregated_score == pytest.approx(0.5 / 61)
    assert by_id["other"].aggregated_score == pytest.approx(0.5 / 61)
