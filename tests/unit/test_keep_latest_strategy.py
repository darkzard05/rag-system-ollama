"""KEEP_LATEST must compare against the first result's timestamp, not wall clock."""

from dataclasses import dataclass

from core.search_aggregator import (
    AggregationStrategy,
    DuplicateStrategy,
    SearchResultAggregator,
)


@dataclass
class _Result:
    doc_id: str
    score: float
    timestamp: float
    node_id: str = "n"
    content: str = ""
    metadata: dict | None = None


def test_keep_latest_selects_newer_result_score() -> None:
    aggregator = SearchResultAggregator(dedup_strategy=DuplicateStrategy.KEEP_LATEST)
    results, _ = aggregator.aggregate_results(
        {
            "n": [
                _Result("d", 5.0, 1000.0),
                _Result("d", 9.0, 1500.0),
            ]
        },
        strategy=AggregationStrategy.DEDUP_ID,
    )
    assert results[0].aggregated_score == 9.0
