"""H2 Recall Blitz — EXTERNAL repro (no src/ edits, no reindex, no model pull).

What it proves:
  (a) config.yml dynamic_top_k.gap_threshold=0.003 is indiscriminate for
      WEIGHTED_RRF aggregated scores (top1 - top10 gap distribution).
  (b) rerank_top_k [5, 8, 12] truncation effect on the dynamic pool
      (min_candidates=12 / max_candidates=18).

Paths:
  MEASURED — pure-python RRF math + (if importable <30s, no network)
             SearchResultAggregator.aggregate_results on synthetic ranks.
             No FAISS index build, no Ollama HTTP.
  SIMULATED — deterministic rank-overlap Monte Carlo for P@1/MRR@k and
             rerank_top_k truncation. Clearly labeled as such.

  Zero Ollama calls by design (all Ollama paths guarded/skipped) so the
  120s Ollama-timeout requirement is vacuously satisfied. Total runtime
  target <180s (typically <10s).

Usage:
    python scripts/evaluation/repro_h2_recall.py [--n 1000] [--seed 42]

Output:
    reports/repro_h2_<timestamp>.md  (+ stdout summary)
Refs:
    config.yml:73-78 (vector_store), :81-102 (retriever/dynamic_top_k),
      :243-261 (grading top_k/grade_top_n),
    src/core/graph/_retrieval.py:286-312 (WEIGHTED_RRF + gap gate),
      :336-353 (rerank_top_k wiring),
    tests/data/golden_set.json:1-26 (5 entries, 3 answerable),
    scripts/evaluation/test_dynamic_topk_gap.py (prior analytical repro).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "data" / "golden_set.json"
CONFIG = ROOT / "config.yml"
REPORTS = ROOT / "reports"

# ---- Config values read from config.yml (verified 2026-09-27) ----
GAP_THRESHOLD = 0.003  # config.yml:91
MIN_CAND = 12  # config.yml:92
MAX_CAND = 18  # config.yml:93
SEARCH_K = 15  # config.yml:84
W_BM25, W_FAISS = 0.4, 0.6  # config.yml:86
GRADING_TOP_K_DEFAULT = 5  # config grading.top_k, config.yml:258
RRF_K = 60  # SearchResultAggregator default


def rrf(rank0: int, k: int = RRF_K) -> float:
    """0-based rank -> 1/(k+rank+1). Matches pure-RRF scale."""
    return 1.0 / (k + rank0 + 1)


def weighted_rrf(rb: int | None, rf: int | None) -> float:
    b = rrf(rb) if rb is not None else 0.0
    f = rrf(rf) if rf is not None else 0.0
    return W_BM25 * b + W_FAISS * f


def load_golden() -> list:
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


def gap_distribution(n: int, seed: int) -> dict:
    """Deterministic overlap Monte Carlo over top-SEARCH_K ranks per engine.

    Realism: FAISS order is a perturbed copy of BM25 order (Kendall-ish
    overlap), so aggregated scores compress exactly like production
    WEIGHTED_RRF over the same k=15 candidate window.
    """
    rng = random.Random(seed)
    gaps: list[float] = []
    triggers = 0
    for _ in range(n):
        bm25_order = list(range(SEARCH_K))
        faiss_order = list(range(SEARCH_K))
        # random adjacent swaps -> partial disagreement, full coverage
        for _ in range(rng.randint(3, 10)):
            i, j = rng.randrange(SEARCH_K), rng.randrange(SEARCH_K)
            faiss_order[i], faiss_order[j] = faiss_order[j], faiss_order[i]
        pos_b = {doc: r for r, doc in enumerate(bm25_order)}
        pos_f = {doc: r for r, doc in enumerate(faiss_order)}
        docs = list(range(SEARCH_K)) + [100 + _]  # 1 faiss-only outsider
        scores = sorted(
            (weighted_rrf(pos_b.get(d), pos_f.get(d)) for d in docs),
            reverse=True,
        )
        gap = scores[0] - scores[9]
        gaps.append(gap)
        if gap > GAP_THRESHOLD:
            triggers += 1
    gaps_sorted = sorted(gaps)
    q = lambda p: gaps_sorted[min(int(p * n), n - 1)]  # noqa: E731
    return {
        "n": n,
        "mean": statistics.fmean(gaps),
        "stdev": statistics.pstdev(gaps),
        "min": min(gaps),
        "max": max(gaps),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "trigger_rate": triggers / n,
        "threshold_x_mean": GAP_THRESHOLD / (statistics.fmean(gaps) or 1e-12),
        "gaps": gaps,
    }


def try_measured_aggregator(timeout_s: float = 30.0) -> dict | None:
    """MEASURED path: real SearchResultAggregator on synthetic ranks.

    No FAISS/Ollama: feeds pre-ranked AggregatedSearchResult lists.
    Guarded by a worker-thread join timeout so a heavy import cannot
    blow the <180s budget.
    """
    import concurrent.futures

    def _work() -> dict:
        sys.path.insert(0, str(ROOT / "src"))
        from types import SimpleNamespace  # stdlib, no src edit

        from core.search_aggregator import (  # type: ignore[import-not-found]
            AggregationStrategy,
            SearchResultAggregator,
        )

        def _item(doc: int, rank: int, node: str) -> SimpleNamespace:
            return SimpleNamespace(
                doc_id=f"d{doc}",
                content=f"c{doc}",
                score=float(SEARCH_K - rank),
                node_id=node,
                metadata={},
            )

        agg = SearchResultAggregator()
        gaps = []
        for trial in range(50):
            docs = list(range(SEARCH_K))
            rng = random.Random(1000 + trial)
            faiss_docs = docs[:]
            rng.shuffle(faiss_docs)
            src = {
                "bm25": [_item(d, r, "bm25") for r, d in enumerate(docs)],
                "faiss": [_item(d, r, "faiss") for r, d in enumerate(faiss_docs)],
            }
            out, _ = agg.aggregate_results(
                src,
                strategy=AggregationStrategy.WEIGHTED_RRF,
                top_k=SEARCH_K,
                weights={"bm25": W_BM25, "faiss": W_FAISS},
            )
            if len(out) >= 10:
                gaps.append(out[0].aggregated_score - out[9].aggregated_score)
        return {
            "n": len(gaps),
            "mean": statistics.fmean(gaps),
            "min": min(gaps),
            "max": max(gaps),
            "trigger_rate": sum(1 for g in gaps if g > GAP_THRESHOLD) / len(gaps),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_work)
        try:
            return fut.result(timeout=timeout_s)
        except Exception as exc:  # import too heavy / API drift -> SIMULATED
            print(f"[measured-path] skipped ({type(exc).__name__}: {exc})")
            return None


def rerank_truncation_effect(gaps: list[float], top_ks=(5, 8, 12)) -> dict:
    """SIMULATED P@1/MRR@k under gold-rank model + truncation math.

    Gold-rank model: answerable golden queries hide gold at rank R with
    P(R=r) ~ geometric(p=0.35) capped at SEARCH_K (optimistic retriever).
    Recall@k = P(R<=k); pool = dynamic_top_k(gap) then truncated to
    rerank_top_k. P@1/MRR reported post-rerank assuming a perfect
    reranker promotes gold to #1 iff gold survives truncation (upper bound).
    """
    import math

    p = 0.35
    probs = []
    rem = 1.0
    for r in range(1, SEARCH_K + 1):
        pr = rem * p if r < SEARCH_K else rem
        probs.append(pr)
        rem -= pr
    trig = sum(1 for g in gaps if g > GAP_THRESHOLD) / len(gaps)
    eff_pool = trig * MIN_CAND + (1 - trig) * MAX_CAND
    rows = {}
    for tk in top_ks:
        # gold survives iff R <= min(pool_for_query, tk); pool ~ MAX_CAND w.h.p.
        # use mixture: trigger queries pool=12, else 18
        rec = 0.0
        mrr = 0.0
        for r, pr in enumerate(probs, start=1):
            surv = trig * (1.0 if r <= min(MIN_CAND, tk) else 0.0) + (1 - trig) * (
                1.0 if r <= min(MAX_CAND, tk) else 0.0
            )
            rec += pr * surv
            mrr += pr * surv * (1.0 / r)  # perfect rerank -> rank 1? no: MRR@k upper
            # NOTE: perfect-rerank upper bound would give 1.0 per survivor;
            # we report rank-discounted MRR (pre-rerank ranks) as conservative LB.
        rows[tk] = {
            "recall_at_k": rec,
            "mrr_conservative": mrr,
            "recall_upper_perfect_rerank": rec,  # same survivors promoted to 1
            "docs_dropped_vs_18": MAX_CAND - tk,
        }
    _ = math  # keep import honest
    return {"trigger_rate": trig, "eff_pool": eff_pool, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.perf_counter()

    golden = load_golden()
    answerable = [g for g in golden if g.get("required_chunks")]
    print(
        f"[golden] entries={len(golden)} answerable={len(answerable)} "
        f"(tests/data/golden_set.json:1-26)"
    )

    sim = gap_distribution(args.n, args.seed)
    print(
        f"[SIMULATED] n={sim['n']} mean_gap={sim['mean']:.6f} "
        f"p50={sim['p50']:.6f} p99={sim['p99']:.6f} "
        f"trigger(>{GAP_THRESHOLD})={sim['trigger_rate']:.1%}"
    )

    measured = try_measured_aggregator(timeout_s=30.0)
    if measured:
        print(
            f"[MEASURED] aggregator n={measured['n']} "
            f"mean={measured['mean']:.6f} trigger={measured['trigger_rate']:.1%}"
        )
    else:
        print("[MEASURED] unavailable -> SIMULATED numbers stand in (labeled).")

    trunc = rerank_truncation_effect(sim["gaps"])
    for tk, row in trunc["rows"].items():
        print(
            f"[rerank_top_k={tk}] recall@{tk}={row['recall_at_k']:.3f} "
            f"MRR_cons={row['mrr_conservative']:.3f} "
            f"dropped_vs_18={row['docs_dropped_vs_18']}"
        )

    # ---- verdict ----
    indiscriminate = sim["trigger_rate"] < 0.05
    verdict = (
        "CONFIRMED: gap_threshold 0.003 is indiscriminate "
        f"(trigger {sim['trigger_rate']:.1%}, threshold ~{sim['threshold_x_mean']:.1f}x mean gap); "
        f"pool almost always {MAX_CAND} (> search_k {SEARCH_K}); "
        f"rerank_top_k={GRADING_TOP_K_DEFAULT} then drops {MAX_CAND - GRADING_TOP_K_DEFAULT} docs."
        if indiscriminate
        else f"NOT confirmed: trigger rate {sim['trigger_rate']:.1%} is discriminative."
    )
    print("[verdict]", verdict)

    elapsed = time.perf_counter() - t0
    REPORTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS / f"repro_h2_{ts}.md"
    lines = [
        "# H2 Recall Blitz — Repro Report",
        "",
        f"- Timestamp: {ts} | runtime {elapsed:.1f}s (budget <180s) | Ollama calls: 0 (guarded/skipped)",
        f"- Golden set: tests/data/golden_set.json — {len(golden)} entries, {len(answerable)} answerable (first 5 read)",
        "- Config refs:",
        "  - config.yml:73-78 vector_store (hnsw_m=32, nprobe=16)",
        "  - config.yml:80-102 retriever (k=15, weights [0.4,0.6], gap_threshold=0.003, min=12, max=18)",
        "  - config.yml:243-261 grading (top_k=5 -> rerank_top_k, grade_top_n=3)",
        "- Code refs:",
        "  - src/core/graph/_retrieval.py:286-291 WEIGHTED_RRF aggregate (top_k=search_k)",
        "  - src/core/graph/_retrieval.py:293-312 gap gate (score_gap = top1-top10)",
        "  - src/core/graph/_retrieval.py:336-353 rerank_top_k wiring + <6char skip",
        "",
        "## 1. Gap distribution (SIMULATED Monte Carlo, deterministic)",
        "",
        f"- n={sim['n']}, seed={args.seed}, mean={sim['mean']:.6f}, stdev={sim['stdev']:.6f}",
        f"- min={sim['min']:.6f}, p50={sim['p50']:.6f}, p90={sim['p90']:.6f}, p99={sim['p99']:.6f}, max={sim['max']:.6f}",
        f"- trigger P(gap > 0.003) = {sim['trigger_rate']:.2%}  (threshold ~= {sim['threshold_x_mean']:.1f}x mean)",
        f"- Verdict: {'INDISCRIMINATE (<5% trigger) — dynamic pool pinned at 18' if indiscriminate else 'discriminative'}",
        "",
        "## 2. Real-aggregator check (MEASURED if available)",
        "",
    ]
    if measured:
        lines += [
            f"- MEASURED via SearchResultAggregator.WEIGHTED_RRF on synthetic ranks: n={measured['n']}",
            f"- mean={measured['mean']:.6f}, min={measured['min']:.6f}, max={measured['max']:.6f}, trigger={measured['trigger_rate']:.2%}",
        ]
    else:
        lines += [
            "- MEASURED path skipped (import/timeout) — SIMULATED numbers above stand in (labeled).",
            "- No FAISS index built, no Ollama call made (heavy-infra fallback per brief).",
        ]
    lines += [
        "",
        "## 3. rerank_top_k truncation (SIMULATED upper-bound model)",
        "",
        f"- Effective pool ~= {trunc['eff_pool']:.2f} (trigger {trunc['trigger_rate']:.2%} -> 12 else 18)",
        "- Gold-rank model: geometric(p=0.35) over top-15; perfect reranker promotes survivors (upper bound).",
        "",
        "| rerank_top_k | recall@k | MRR conservative | dropped vs 18 |",
        "|---|---|---|---|",
    ]
    for tk, row in trunc["rows"].items():
        lines.append(
            f"| {tk} | {row['recall_at_k']:.3f} | {row['mrr_conservative']:.3f} | {row['docs_dropped_vs_18']} |"
        )
    lines += [
        "",
        "## 4. Retrieval-config comparison",
        "",
        "| config | pool | rerank_top_k | recall@k (sim) | note |",
        "|---|---|---|---|---|",
        f"| current (gap 0.003, max 18, top_k 5) | ~{trunc['eff_pool']:.1f} | 5 | {trunc['rows'][5]['recall_at_k']:.3f} | baseline: 13 docs reranked then dropped |",
        f"| top_k 8 | ~{trunc['eff_pool']:.1f} | 8 | {trunc['rows'][8]['recall_at_k']:.3f} | +{trunc['rows'][8]['recall_at_k'] - trunc['rows'][5]['recall_at_k']:.3f} recall vs 5 |",
        f"| top_k 12 (=min_candidates) | ~{trunc['eff_pool']:.1f} | 12 | {trunc['rows'][12]['recall_at_k']:.3f} | +{trunc['rows'][12]['recall_at_k'] - trunc['rows'][5]['recall_at_k']:.3f} recall vs 5 |",
        "| calibrated gap ~0.001 (pool~12 discriminative) + top_k 8 | ~12 | 8 | — (needs live FAISS) | recommended next MEASURED test, no src edit |",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "_Labels: SIMULATED = deterministic math model; MEASURED = real SearchResultAggregator code path. No src/ or config.yml edits; no reindex; no model pull._",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {out} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
