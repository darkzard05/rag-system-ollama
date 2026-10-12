"""Cache-key normalization harness (M3 direct evidence, offline).

Measures, with a counting fake embedder (no model/Ollama):
1. MemoizingEmbedding whitespace-variant hit rate + underlying call count.
2. AsyncSemanticReranker query-cache model-namespacing (same text, other
   model must NOT reuse a stale-dim vector).

Usage:
    python scripts/bench_cache_hit.py [--repeat 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"


class _CountingEmbedder:
    def __init__(self, dim: int = 8, tag: str = "fake") -> None:
        self.calls = 0
        self._dim = dim
        self.model = tag

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        base = float(sum(ord(c) for c in text) % 97)
        return [base + i for i in range(self._dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


async def main_async(repeat: int) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from langchain_core.documents import Document

    from core.async_reranker import AsyncSemanticReranker
    from core.model_loader import MemoizingEmbedding

    # --- 1. whitespace-variant hit rate ---
    inner = _CountingEmbedder()
    memo = MemoizingEmbedding(inner)
    variants = [
        " attention mechanism 이란 ",
        "attention  mechanism 이란",
        "attention mechanism 이란",
    ]
    t0 = time.perf_counter()
    for _ in range(repeat):
        for v in variants:
            memo.embed_query(v)
    memo_ms = (time.perf_counter() - t0) * 1000.0
    stats = memo.stats()

    # --- 2. model-namespacing (stale-dim guard) ---
    emb_a = _CountingEmbedder(dim=8, tag="model-A")
    emb_b = _CountingEmbedder(dim=16, tag="model-B")
    ranker_b = AsyncSemanticReranker(embedder=emb_b)  # type: ignore[arg-type]
    docs = [
        Document(page_content="alpha attention mechanism", metadata={}),
        Document(page_content="beta cooking recipe", metadata={}),
    ]
    # Prime model-A cache shape via a ranker on A, then query B with same text.
    ranker_a = AsyncSemanticReranker(embedder=emb_a)  # type: ignore[arg-type]
    await ranker_a.rerank(docs, "attention mechanism", top_k=2)
    ranked, scores = await ranker_b.rerank(
        [Document(page_content="alpha attention mechanism", metadata={})],
        "attention mechanism",
        top_k=1,
    )
    namespaced_ok = (
        emb_b.calls >= 1  # B computed its own dim-16 vector, not reused dim-8
        and len(scores) == 1
        and ranked[0].metadata.get("rerank_score") is not None
    )

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "repeat": repeat,
        "memo": {
            "variants": variants,
            "calls": inner.calls,
            "stats": stats,
            "total_ms": memo_ms,
            # 3 variants x repeat calls; ideal = 1 underlying call.
            "hit_rate": stats["hits"] / max(1, stats["hits"] + stats["misses"]),
        },
        "reranker_namespacing": {
            "model_b_calls": emb_b.calls,
            "ok": bool(namespaced_ok),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    report = asyncio.run(main_async(args.repeat))
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"cache_hit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    m = report["memo"]
    print(
        f"[cache] memo calls={m['calls']} hits={m['stats']['hits']} "
        f"misses={m['stats']['misses']} hit_rate={m['hit_rate']:.2f} "
        f"({m['total_ms']:.1f}ms)"
    )
    print(
        f"[cache] reranker model-namespacing ok={report['reranker_namespacing']['ok']} "
        f"(model-B calls={report['reranker_namespacing']['model_b_calls']})"
    )
    print(f"[cache] report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
