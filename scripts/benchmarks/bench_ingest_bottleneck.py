"""Measure cold document ingest, including the tokenizer load on the build path.

Usage:
    python scripts/benchmarks/bench_ingest_bottleneck.py --label before

Each invocation runs in a fresh process so the tokenizer import is not warm.
The script parses the existing ``[INDEX][TIMING]`` log line and reports stage
times plus wall time. It writes one JSON artifact under ``reports/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.config import DEFAULT_EMBEDDING_MODEL, ENABLE_VECTOR_CACHE  # noqa: E402
from common.logging_config import setup_logging  # noqa: E402
from core.model_loader import ModelManager  # noqa: E402
from core.rag_core import RAGSystem  # noqa: E402
from core.session import SessionManager  # noqa: E402

_STAGE_RE = re.compile(
    r"extract_ms=(?P<extract_ms>[\d.]+) chunk_ms=(?P<chunk_ms>[\d.]+) "
    r"faiss_ms=(?P<faiss_ms>[\d.]+) bm25_ms=(?P<bm25_ms>[\d.]+) "
    r"register_ms=(?P<register_ms>[\d.]+) total_s=(?P<total_s>[\d.]+) "
    r"chunks=(?P<chunks>\d+)"
)
_WARMUP_RE = re.compile(r"토크나이저 워밍업 \| loaded=\S+ \| elapsed_ms=(?P<ms>[\d.]+)")


class _TimingCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.stages: dict[str, float] = {}
        self.tokenizer_warmup_ms = 0.0
        self.chunks = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        warmup = _WARMUP_RE.search(message)
        if warmup:
            self.tokenizer_warmup_ms = float(warmup.group("ms"))
        match = _STAGE_RE.search(message)
        if not match:
            return
        for key in (
            "extract_ms",
            "chunk_ms",
            "faiss_ms",
            "bm25_ms",
            "register_ms",
            "total_s",
        ):
            self.stages[key] = float(match.group(key))
        self.chunks = int(match.group("chunks"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="tests/data/2201.07520v1.pdf")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.is_absolute():
        pdf = ROOT / pdf
    if not pdf.exists():
        print(f"[ERROR] missing pdf: {pdf}", file=sys.stderr)
        return 1

    setup_logging(log_level="INFO")
    capture = _TimingCapture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    session_id = f"bench-ingest-{int(datetime.now().timestamp() * 1000)}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    embedder = asyncio.run(ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL))
    started = time.perf_counter()
    asyncio.run(rag.build_pipeline(str(pdf), pdf.name, embedder))
    wall_s = time.perf_counter() - started

    if not capture.stages:
        print("[ERROR] no [INDEX][TIMING] record captured", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "pdf": str(pdf),
        "embedder": DEFAULT_EMBEDDING_MODEL,
        "vector_cache_enabled": bool(ENABLE_VECTOR_CACHE),
        "wall_s": round(wall_s, 3),
        "tokenizer_warmup_ms": round(capture.tokenizer_warmup_ms, 1),
        "stages": {key: round(value, 1) for key, value in capture.stages.items()},
        "chunks": capture.chunks,
        "errors": 0,
    }

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    out = (
        Path(args.out)
        if args.out
        else reports / f"bench_ingest_{args.label}_{int(time.time())}.json"
    )
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
