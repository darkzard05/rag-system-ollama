"""FlashRank rerank cache benchmark, offline (no Ollama).

Uses the real cached PDF candidate pool and the configured FlashRank engine.
For each pair, the first call starts from an empty score cache and the second
reuses the same query and the same candidate documents. The script records
both timings and checks that the two calls return the same order and scores.

Usage:
    python scripts/benchmarks/bench_rerank_cache.py --repeat 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import benchmark_reranker as rerank_bench  # noqa: E402


def _copy_docs(docs: list[Any]) -> list[Any]:
    from langchain_core.documents import Document

    return [
        Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs
    ]


async def _run(repeat: int) -> dict[str, Any]:
    from core.async_reranker import get_async_reranker

    pool, pool_type, pool_errors = rerank_bench._build_pool()
    ranker = await get_async_reranker()
    clear_cache = getattr(ranker, "clear_score_cache", None)

    cold_ms: list[float] = []
    warm_ms: list[float] = []
    mismatch: list[int] = []
    for i in range(repeat):
        if clear_cache is not None:
            clear_cache()
        cold_docs = _copy_docs(pool)
        start = time.perf_counter()
        cold_docs, cold_scores = await ranker.rerank(
            cold_docs, query=rerank_bench.QUERY, top_k=len(pool)
        )
        cold_ms.append((time.perf_counter() - start) * 1000.0)

        warm_docs = _copy_docs(pool)
        start = time.perf_counter()
        warm_docs, warm_scores = await ranker.rerank(
            warm_docs, query=rerank_bench.QUERY, top_k=len(pool)
        )
        warm_ms.append((time.perf_counter() - start) * 1000.0)

        cold_ids = [d.page_content for d in cold_docs]
        warm_ids = [d.page_content for d in warm_docs]
        if cold_ids != warm_ids or any(
            abs(a - b) > 1e-6 for a, b in zip(cold_scores, warm_scores, strict=True)
        ):
            mismatch.append(i)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool_type": pool_type,
        "candidates": len(pool),
        "repeat": repeat,
        "cold_ms": [round(v, 2) for v in cold_ms],
        "warm_ms": [round(v, 2) for v in warm_ms],
        "cold_median_ms": round(statistics.median(cold_ms), 2),
        "warm_median_ms": round(statistics.median(warm_ms), 2),
        "cold_range_ms": [round(min(cold_ms), 2), round(max(cold_ms), 2)],
        "warm_range_ms": [round(min(warm_ms), 2), round(max(warm_ms), 2)],
        "mismatch_pairs": mismatch,
        "errors": list(pool_errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    report = asyncio.run(_run(args.repeat))

    out = (
        Path(args.out)
        if args.out
        else ROOT
        / "reports"
        / (f"rerank_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[rerank] cold median={report['cold_median_ms']:.1f} ms "
        f"range={report['cold_range_ms']}"
    )
    print(
        f"[rerank] warm median={report['warm_median_ms']:.1f} ms "
        f"range={report['warm_range_ms']}"
    )
    print(f"[rerank] mismatch_pairs={report['mismatch_pairs']} report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
