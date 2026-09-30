"""
H2 Reproduction: Dynamic top_k gap_threshold 0.003 is miscalibrated for
weighted RRF, making the mechanism effectively dead.

The config sets dynamic_top_k.gap_threshold: 0.003. The code in
_retrieval.py computes score_gap = top_1_score - top_10_score from
aggregated WEIGHTED_RRF results. But the gap_threshold was designed
for pure RRF scores (1/(k+rank)), not weighted combinations of BM25
and FAISS scores.

With ensemble_weights [0.4, 0.6], the weighted RRF compresses the
score distribution, making gaps smaller than 0.003 for most queries.
This means score_gap > gap_threshold is almost never true, and
dynamic_top_k always falls back to max_candidates: 18.

USAGE:
    python scripts/evaluation/test_dynamic_topk_gap.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np


def compute_rrf_score(rank: int, k: int = 60) -> float:
    """Pure RRF score: 1/(k+rank)"""
    return 1.0 / (k + rank)


def compute_weighted_rrf(
    bm25_rank: int,
    faiss_rank: int,
    bm25_weight: float = 0.4,
    faiss_weight: float = 0.6,
    k: int = 60,
) -> float:
    """Weighted RRF as used in the system."""
    bm25_rrf = compute_rrf_score(bm25_rank, k) if bm25_rank >= 0 else 0.0
    faiss_rrf = compute_rrf_score(faiss_rank, k) if faiss_rank >= 0 else 0.0
    return bm25_weight * bm25_rrf + faiss_weight * faiss_rrf


def main():
    print("=" * 70)
    print("H2: Dynamic Top-K Gap Threshold Calibration Test")
    print("=" * 70)

    gap_threshold = 0.003
    min_candidates = 12
    max_candidates = 18
    search_k = 15  # From config.yml retriever.search_kwargs.k

    print(f"\nConfig: gap_threshold={gap_threshold}, min_candidates={min_candidates},")
    print(f"        max_candidates={max_candidates}, search_k={search_k}")
    print(f"        ensemble_weights=[0.4, 0.6]")

    # Simulate 100 queries with realistic rank distributions
    np.random.seed(42)
    n_queries = 100

    gaps_pure_rrf = []
    gaps_weighted_rrf = []

    for _ in range(n_queries):
        # Simulate BM25 and FAISS rankings (some overlap, some different)
        # Top 15 from each engine
        bm25_ranks = list(range(15))
        faiss_ranks = list(range(15))
        np.random.shuffle(faiss_ranks)

        # Compute weighted RRF scores for top 15 aggregated results
        scores = []
        for rank in range(1, 16):
            # Find BM25 rank for this position
            bm25_pos = bm25_ranks.index(rank - 1) if (rank - 1) in bm25_ranks else 15
            faiss_pos = faiss_ranks.index(rank - 1) if (rank - 1) in faiss_ranks else 15
            score = compute_weighted_rrf(bm25_pos, faiss_pos)
            scores.append(score)

        scores.sort(reverse=True)

        if len(scores) >= 10:
            gap_pure = compute_rrf_score(1) - compute_rrf_score(10)
            gap_weighted = scores[0] - scores[9]
            gaps_pure_rrf.append(gap_pure)
            gaps_weighted_rrf.append(gap_weighted)

    gaps_pure_rrf = np.array(gaps_pure_rrf)
    gaps_weighted_rrf = np.array(gaps_weighted_rrf)

    print(f"\n--- Pure RRF (k=60) gap distribution ---")
    print(f"Mean gap: {gaps_pure_rrf.mean():.6f}")
    print(f"Min gap: {gaps_pure_rrf.min():.6f}")
    print(f"Max gap: {gaps_pure_rrf.max():.6f}")
    print(f"Gap > {gap_threshold}: {(gaps_pure_rrf > gap_threshold).mean():.1%}")

    print(f"\n--- Weighted RRF (0.4/0.6) gap distribution ---")
    print(f"Mean gap: {gaps_weighted_rrf.mean():.6f}")
    print(f"Min gap: {gaps_weighted_rrf.min():.6f}")
    print(f"Max gap: {gaps_weighted_rrf.max():.6f}")
    print(f"Gap > {gap_threshold}: {(gaps_weighted_rrf > gap_threshold).mean():.1%}")

    # Show the problem
    print(f"\n--- Problem Analysis ---")
    print(
        f"With weighted RRF, only {(gaps_weighted_rrf > gap_threshold).mean():.1%} of queries"
    )
    print(f"trigger dynamic_top_k=min_candidates={min_candidates}.")
    print(
        f"{(gaps_weighted_rrf <= gap_threshold).mean():.1%} of queries use max_candidates={max_candidates}."
    )
    print(f"Since max_candidates ({max_candidates}) > search_k ({search_k}),")
    print(
        f"the system pulls in {max_candidates - search_k} extra low-quality candidates"
    )
    print(f"that must be reranked, increasing latency without quality gain.")

    # Find the right threshold
    print(f"\n--- Recommended Calibration ---")
    print(
        f"For weighted RRF, gap_threshold should be ~{gaps_weighted_rrf.mean() * 0.5:.6f}"
    )
    print(f"(approximately 50% of mean gap)")
    print(
        f"Current gap_threshold={gap_threshold} is {gap_threshold / gaps_weighted_rrf.mean():.1f}x the mean gap"
    )
    print(f"RECOMMENDATION: Set gap_threshold to ~0.001 or lower, or use")
    print(f"percentile-based threshold on the actual gap distribution.")
    print("=" * 70)


if __name__ == "__main__":
    main()
