"""RRF parity check: builtin SearchResultAggregator vs reference formula.

Reference mirrors langchain-classic EnsembleRetriever.weighted_reciprocal_rank:
    score(doc) = sum(weight[nid] / (c + rank_nid(doc)))  (1-based rank)

Run: python scripts/bench_rrf_parity.py [--c 60]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass


@dataclass
class _FakeResult:
    doc_id: str
    content: str
    score: float
    node_id: str = "unknown"
    metadata: dict | None = None


def _reference_rrf(
    search_results: dict[str, list[_FakeResult]],
    c: int = 60,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    w = weights or dict.fromkeys(search_results, 1.0)
    for nid, results in search_results.items():
        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(ranked, start=1):
            out[r.doc_id] = out.get(r.doc_id, 0.0) + w.get(nid, 1.0) / (c + rank)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, default=60)
    args = parser.parse_args()

    sys.path.insert(0, "src")
    from core.search_aggregator import (
        AggregationStrategy,
        SearchResultAggregator,
    )

    # Synthetic 2-node corpus (BM25 + FAISS shape).
    bm25 = [
        _FakeResult(f"doc{i}", f"content {i}", score=10.0 - i, node_id="bm25")
        for i in range(8)
    ]
    faiss_unsorted = [
        _FakeResult(
            f"doc{i}", f"content {i}", score=0.9 - 0.05 * abs(3 - i), node_id="faiss"
        )
        for i in reversed(range(8))
    ]
    # NOTE: builtin _rrf_fusion_2node contracts input order == rank order
    # (production callers already return score-descending). Reference sorts
    # internally, so parity requires pre-sorted inputs here.
    faiss = sorted(faiss_unsorted, key=lambda r: r.score, reverse=True)
    # Overlap is partial on purpose: doc0..7 in both, plus unique tails.
    bm25.append(_FakeResult("bm25-only", "unique bm25", score=0.5, node_id="bm25"))
    faiss.append(_FakeResult("faiss-only", "unique faiss", score=0.5, node_id="faiss"))
    # Keep descending order after appending tails (0.5 < all real scores).
    bm25 = sorted(bm25, key=lambda r: r.score, reverse=True)
    faiss = sorted(faiss, key=lambda r: r.score, reverse=True)
    inputs: dict[str, list[_FakeResult]] = {"bm25": bm25, "faiss": faiss}
    weights = {"bm25": 0.4, "faiss": 0.6}

    agg = SearchResultAggregator()
    t0 = time.perf_counter()
    got, metrics = agg.aggregate_results(
        inputs,  # type: ignore[arg-type]
        strategy=AggregationStrategy.WEIGHTED_RRF,
        weights=weights,
    )
    # Inject c parity: builtin defaults k=60; reference uses --c.
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    expected = _reference_rrf(inputs, c=args.c, weights=weights)
    got_map = {r.doc_id: r.aggregated_score for r in got}

    mismatches = []
    for doc_id, exp in expected.items():
        val = got_map.get(doc_id)
        if val is None or abs(val - exp) > 1e-9:
            mismatches.append((doc_id, exp, val))

    print(f"[RRF parity] c={args.c} docs={len(expected)} time={elapsed_ms:.2f}ms")
    print(
        f"[RRF parity] strategy={metrics.strategy_used} inputs={metrics.total_input_results}"
    )
    if mismatches:
        print(f"[RRF parity] FAIL {len(mismatches)} mismatches:")
        for doc_id, exp, val in mismatches[:10]:
            print(f"  {doc_id}: expected={exp:.9f} got={val}")
        return 1
    top = sorted(got, key=lambda r: r.aggregated_score, reverse=True)[:3]
    for r in top:
        print(f"  top {r.doc_id}={r.aggregated_score:.6f} sources={r.source_nodes}")
    print("[RRF parity] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
