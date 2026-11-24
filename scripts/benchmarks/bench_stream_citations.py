"""Rerunnable microbenchmark for the Python streaming citation path.

Times the work done on each incremental ``citations_add`` event and on the
final citation batch. Uses the real merge and hydration functions. No LLM,
no Streamlit server, no PDF.

Usage:
    python scripts/benchmarks/bench_stream_citations.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

N_DOCS = 12
N_CITES = 6
TEXT_SIZE = 2000
ANCHOR_SIZE = 100
TRIALS = 7
INNER_REPEATS = 100


def _docs(*, explicit_id: bool) -> list[Any]:
    docs = []
    for i in range(N_DOCS):
        metadata: dict[str, Any] = {
            "page": (i % 5) + 1,
            "current_section": f"Section {i % 3}",
            "rerank_score": 0.9 - i * 0.01,
        }
        if explicit_id:
            metadata["doc_id"] = f"doc-{i}"
        docs.append(
            SimpleNamespace(
                page_content=f"tail benchmark doc {i} " + "alpha " * (TEXT_SIZE // 6),
                metadata=metadata,
            )
        )
    return docs


def _citations(doc_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": doc_ids[i % N_DOCS],
            "text_span": f"span {i} " + "beta " * (ANCHOR_SIZE // 5),
            "section": f"Section {i % 3}",
            "page": (i % 5) + 1,
            "score": 0.9,
        }
        for i in range(N_CITES)
    ]


def _one_scenario(*, explicit_id: bool) -> float:
    from common.utils import doc_stable_id
    from core.graph._generate import _enrich_citations, _filter_stream_cite_items
    from ui.components.streaming_core import merge_citations_add

    docs = _docs(explicit_id=explicit_id)
    doc_ids = [doc_stable_id(d) for d in docs]
    cites = _citations(doc_ids)
    valid_ids = set(doc_ids)

    start = time.perf_counter()
    merged: list[dict[str, Any]] = []
    for i, item in enumerate(cites):
        merged = merge_citations_add(
            merged,
            items=[item],
            base_index=i,
            arr_len=i + 1,
            valid_ids=valid_ids if explicit_id else None,
        )
    enriched = _enrich_citations(_filter_stream_cite_items(cites, docs), docs)
    assert len(merged) == N_CITES
    assert len(enriched) == N_CITES
    return (time.perf_counter() - start) * 1000


def _run(*, explicit_id: bool) -> dict[str, float]:
    samples = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        for _ in range(INNER_REPEATS):
            _one_scenario(explicit_id=explicit_id)
        samples.append((time.perf_counter() - t0) * 1000 / INNER_REPEATS)
    return {
        "median_ms_per_answer": statistics.median(samples),
        "min_ms_per_answer": min(samples),
        "max_ms_per_answer": max(samples),
        "trials": TRIALS,
    }


def main() -> None:
    result = {
        "tool": "bench_stream_citations",
        "n_docs": N_DOCS,
        "n_citations": N_CITES,
        "explicit_doc_id": _run(explicit_id=True),
        "hashed_doc_id": _run(explicit_id=False),
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports / f"stream_citation_bench_{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
