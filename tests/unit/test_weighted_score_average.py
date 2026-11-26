"""WEIGHTED_SCORE must converge to the arithmetic mean of merged scores."""

from dataclasses import dataclass

from core.search_aggregator import AggregationStrategy, SearchResultAggregator


@dataclass
class _Result:
    doc_id: str
    score: float
    node_id: str = "n"
    content: str = ""
    metadata: dict | None = None


def test_three_scores_produce_correct_running_mean() -> None:
    results, _ = SearchResultAggregator().aggregate_results(
        {"n": [_Result("d", 10.0), _Result("d", 8.0), _Result("d", 6.0)]},
        strategy=AggregationStrategy.WEIGHTED_SCORE,
        weights={"n": 1.0},
    )
    assert len(results) == 1
    assert results[0].aggregated_score == 8.0
