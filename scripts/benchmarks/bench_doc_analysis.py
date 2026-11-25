"""Measure the document-analysis (ingest) stages and token-count work.

Usage:
    python scripts/benchmarks/bench_doc_analysis.py --pdf tests/data/2201.07520v1.pdf

Reports per stage wall time and the amount of token-counting work the chunking
path performs, so redundant recomputation shows up as a number instead of a
guess. Writes JSON to reports/.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))


@dataclass
class CountCall:
    texts: int = 0
    chars: int = 0
    calls: int = 0
    cpu_s: float = 0.0


@dataclass
class Run:
    label: str
    stage_s: dict[str, float] = field(default_factory=dict)
    counter: CountCall = field(default_factory=CountCall)
    chunks: int = 0
    pages: int = 0


class FakeEmbedder:
    """Deterministic in-process embedder; removes HTTP noise from the measurement."""

    def __init__(self, dim: int = 64) -> None:
        self.model = "bench-fake-embedder"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        raw = [(b - 127.5) / 127.5 for b in digest]
        out: list[float] = []
        while len(out) < self.dim:
            out.extend(raw)
        return out[: self.dim]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _instrument_token_counter() -> tuple[Any, Any]:
    """Wrap the chunking token counter and record work + CPU time."""
    from core import chunking

    original = chunking._embedding_token_counts
    state = CountCall()

    def wrapper(texts: list[str]) -> list[int]:
        state.calls += 1
        state.texts += len(texts)
        state.chars += sum(len(t) for t in texts)
        t0 = time.process_time()
        try:
            return original(texts)
        finally:
            state.cpu_s += time.process_time() - t0

    chunking._embedding_token_counts = wrapper  # type: ignore[assignment]
    return state, original


def _rough_work(docs: list[Any]) -> float:
    """CPU seconds of one full count_tokens_rough pass over the corpus."""
    from common.text_utils import count_tokens_rough

    t0 = time.process_time()
    for doc in docs:
        count_tokens_rough(doc.page_content)
    return time.process_time() - t0


async def run_once(pdf: str, use_ollama: bool) -> Run:
    from core.document_processor import load_pdf_docs

    result = Run(label="ollama" if use_ollama else "fake")

    t0 = time.perf_counter()
    docs = await load_pdf_docs(pdf, os.path.basename(pdf), session_id="bench-doc")
    result.stage_s["extract"] = time.perf_counter() - t0
    result.pages = len(docs)

    if use_ollama:
        from core.model_loader import ModelManager

        embedder: Any = await ModelManager.get_embedder()
    else:
        embedder = FakeEmbedder()

    state, original = _instrument_token_counter()
    try:
        from core.chunking import split_documents

        t0 = time.perf_counter()
        chunks, _vectors = await split_documents(docs, embedder=embedder)
        result.stage_s["chunk"] = time.perf_counter() - t0
        result.chunks = len(chunks)
    finally:
        from core import chunking

        chunking._embedding_token_counts = original  # type: ignore[assignment]

    result.counter = state
    result.stage_s["rough_pass_all_pages"] = _rough_work(docs)
    result.stage_s["rough_pass_all_chunks"] = _rough_work(chunks)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf", default=os.path.join("tests", "data", "2201.07520v1.pdf")
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--embedder", choices=("fake", "ollama"), default="fake")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    pdf = args.pdf if os.path.isabs(args.pdf) else os.path.join(ROOT, args.pdf)
    if not os.path.exists(pdf):
        print(f"[ERROR] missing pdf: {pdf}")
        return 1

    import asyncio

    runs: list[Run] = []
    for i in range(args.runs):
        runs.append(asyncio.run(run_once(pdf, args.embedder == "ollama")))
        r = runs[-1]
        print(
            f"[run {i + 1}/{args.runs}] pages={r.pages} chunks={r.chunks} "
            f"extract={r.stage_s['extract'] * 1000:.0f}ms "
            f"chunk={r.stage_s['chunk'] * 1000:.0f}ms "
            f"token_calls={r.counter.calls} texts={r.counter.texts} "
            f"token_cpu={r.counter.cpu_s * 1000:.0f}ms"
        )

    def med(key: str) -> float:
        return statistics.median(r.stage_s[key] for r in runs)

    summary = {
        "label": args.label,
        "pdf": args.pdf,
        "embedder": args.embedder,
        "runs": len(runs),
        "pages": runs[0].pages,
        "chunks": runs[0].chunks,
        "median": {
            "extract_ms": med("extract") * 1000,
            "chunk_ms": med("chunk") * 1000,
            "token_count_cpu_ms": statistics.median(r.counter.cpu_s for r in runs)
            * 1000,
            "rough_pass_pages_ms": med("rough_pass_all_pages") * 1000,
            "rough_pass_chunks_ms": med("rough_pass_all_chunks") * 1000,
        },
        "runs_detail": [
            {
                "stage_ms": {k: v * 1000 for k, v in r.stage_s.items()},
                "token_calls": r.counter.calls,
                "token_texts": r.counter.texts,
                "token_chars": r.counter.chars,
                "token_cpu_ms": r.counter.cpu_s * 1000,
            }
            for r in runs
        ],
    }

    out_dir = os.path.join(ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, f"bench_doc_analysis_{args.label}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n=== median ===")
    for key, value in summary["median"].items():
        print(f"{key:>24}: {value:.1f}")
    print(f"artifact: {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
