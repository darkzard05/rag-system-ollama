"""Streaming-complete -> citation-painted gap measurement.

Covers the blind spot of existing harnesses:
- ``measure_tail.py`` uses synthetic dict docs (no real fitz cost).
- ``verify_citation_tail.py`` measures mid-stream P0, not the post-stream
  finalize gap (join-block -> live persist -> finalize -> pdf finalize).

Modes:
    python scripts/measure_stream_to_citation_gap.py --fitz-bench
    python scripts/measure_stream_to_citation_gap.py --log path/to/app.log
    python scripts/measure_stream_to_citation_gap.py --fitz-bench --log path/to/app.log

--fitz-bench builds realistic docs pointing at tests/data/2201.07520v1.pdf
with NO precomputed coords (on-demand fitz path, i.e. prewarm miss) and
times ``extract_annotations_from_docs`` cold vs warm(memo) plus
``ordered_reference_entries``. No Ollama/LLM needed.

--log parses existing PERF lines and prints a gap table + verdict.
Writes reports/stream_gap_<ts>.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
TEST_PDF = REPO_ROOT / "tests" / "data" / "2201.07520v1.pdf"

sys.path.insert(0, str(REPO_ROOT / "src"))

SPAN_RE = re.compile(r"\[PERF\] (.+?) took ([0-9.]+)s(.*)")
PDF_META_RE = re.compile(
    r"pdf finalize took .*?cache_hit=(\w+).*?prewarmed=(\w+)", re.IGNORECASE
)
TAIL_RE = re.compile(r"tail_content_ms=([0-9.\-]+)")
PARSE_RE = re.compile(r"parse_ms=([0-9.\-]+)")
CIT_COUNT_RE = re.compile(r"citations-dispatch count=(\d+)")
HYDRATION_RE = re.compile(r"hydration drain.*?([0-9.]+)s", re.IGNORECASE)

VERDICT_THRESHOLDS = {
    "pdf_finalize_s": 0.5,
    "hydration_s": 1.0,
    "tail_content_ms": 2000.0,
    "recompute_s": 0.3,
    "persist_s": 0.5,
}


def parse_log_file(path: Path) -> dict[str, Any]:
    """Parse PERF/tail lines from a log file into a gap summary."""
    text = path.read_text(encoding="utf-8", errors="replace")
    spans: dict[str, list[float]] = {}
    pdf_flags: list[dict[str, str]] = []
    tails: list[float] = []
    parses: list[float] = []
    cit_counts: list[int] = []
    for line in text.splitlines():
        m = SPAN_RE.search(line)
        if m:
            name = m.group(1).strip()
            try:
                spans.setdefault(name, []).append(float(m.group(2)))
            except ValueError:
                continue
            pm = PDF_META_RE.search(line)
            if pm:
                pdf_flags.append({"cache_hit": pm.group(1), "prewarmed": pm.group(2)})
        tm = TAIL_RE.search(line)
        if tm:
            with contextlib.suppress(ValueError):
                tails.append(float(tm.group(1)))
        pm2 = PARSE_RE.search(line)
        if pm2:
            with contextlib.suppress(ValueError):
                parses.append(float(pm2.group(1)))
        cm = CIT_COUNT_RE.search(line)
        if cm:
            cit_counts.append(int(cm.group(1)))

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"n": 0, "mean": 0.0, "max": 0.0}
        return {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "max": max(vals),
        }

    return {
        "spans": {k: _stats(v) for k, v in spans.items()},
        "pdf_flags": pdf_flags,
        "tail_content_ms": _stats(tails),
        "parse_ms": _stats(parses),
        "citation_counts": cit_counts,
    }


def _pdf_sentence_pool() -> list[tuple[str, int]]:
    """Real sentences from the test PDF so fitz actually matches boxes."""
    try:
        import pymupdf  # type: ignore[import-not-found]

        doc = pymupdf.open(str(TEST_PDF))
    except Exception:
        try:
            import fitz  # type: ignore[import-not-found]

            doc = fitz.open(str(TEST_PDF))
        except Exception:
            return []
    pool: list[tuple[str, int]] = []
    for pno in range(min(len(doc), 6)):
        text = str(doc[pno].get_text())
        for line in text.splitlines():
            s = " ".join(line.split()).strip()
            if len(s) >= 40 and len(s) <= 200 and s.count(" ") >= 5:
                pool.append((s, pno + 1))
                if len(pool) >= 24:
                    break
        if len(pool) >= 24:
            break
    with contextlib.suppress(Exception):
        doc.close()
    return pool


def _realistic_docs(n: int = 6) -> list[dict[str, Any]]:
    """Docs with real file_path but no coords -> on-demand fitz path."""
    pool = _pdf_sentence_pool()
    docs: list[dict[str, Any]] = []
    for i in range(n):
        if pool:
            content, page = pool[i % len(pool)]
        else:
            content = (
                "causally masked objective VQVAE-GAN tokens sequence "
                f"modeling evidence span doc {i} "
            )
            page = (i % 4) + 1
        docs.append(
            {
                "page_content": content,
                "metadata": {
                    "file_path": str(TEST_PDF),
                    "source": str(TEST_PDF),
                    "page": page,
                    "current_section": f"Section {i % 3}",
                },
            }
        )
    return docs


def _realistic_citations(n: int = 6) -> list[dict[str, Any]]:
    pool = _pdf_sentence_pool()
    cites: list[dict[str, Any]] = []
    for i in range(n):
        if pool:
            span, page = pool[i % len(pool)]
            cites.append(
                {
                    "doc_id": i,
                    "text_span": span[:200],
                    "section": f"Section {i % 3}",
                    "page": page,
                    "score": 0.9,
                }
            )
        else:
            cites.append(
                {
                    "doc_id": i,
                    "text_span": f"evidence span doc {i}",
                    "section": f"Section {i % 3}",
                    "page": (i % 4) + 1,
                    "score": 0.9,
                }
            )
    return cites


def run_fitz_bench(n_docs: int = 6) -> dict[str, Any]:
    """Time real fitz finalize path cold (miss) vs warm (memo)."""
    from common.utils import extract_annotations_from_docs

    try:
        from ui.components.chat_references import ordered_reference_entries
    except Exception:
        ordered_reference_entries = None  # type: ignore[assignment]

    if not TEST_PDF.exists():
        return {"skipped": f"test pdf not found: {TEST_PDF}"}

    docs = _realistic_docs(n_docs)
    citations = _realistic_citations(n_docs)

    t0 = time.perf_counter()
    ann_cold = extract_annotations_from_docs(docs, citations)
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ann_warm = extract_annotations_from_docs(docs, citations)
    warm_s = time.perf_counter() - t0

    ref_s = -1.0
    n_entries = -1
    if ordered_reference_entries is not None:
        t0 = time.perf_counter()
        entries = ordered_reference_entries(docs, citations)
        ref_s = time.perf_counter() - t0
        n_entries = len(entries)

    return {
        "n_docs": n_docs,
        "cold_s": cold_s,
        "warm_s": warm_s,
        "cold_annotations": len(ann_cold),
        "warm_annotations": len(ann_warm),
        "recompute_s": ref_s,
        "n_entries": n_entries,
    }


def verdict(log_summary: dict[str, Any] | None, fitz: dict[str, Any] | None) -> str:
    """Single-line bottleneck verdict from thresholds."""
    suspects: list[str] = []
    if (
        fitz
        and "cold_s" in fitz
        and float(fitz["cold_s"]) >= VERDICT_THRESHOLDS["pdf_finalize_s"]
    ):
        suspects.append(
            f"PDF동기(fitz cold {fitz['cold_s']:.2f}s, "
            f"warm {fitz.get('warm_s', 0):.2f}s)"
        )
    if log_summary:
        spans = log_summary.get("spans", {})

        def _max(key_part: str) -> float:
            best = 0.0
            for k, v in spans.items():
                if key_part in k:
                    best = max(best, float(v.get("max", 0.0)))
            return best

        if _max("hydration") >= VERDICT_THRESHOLDS["hydration_s"]:
            suspects.append(f"hydration drain({_max('hydration'):.2f}s)")
        if _max("pdf finalize") >= VERDICT_THRESHOLDS["pdf_finalize_s"]:
            suspects.append(f"pdf finalize({_max('pdf finalize'):.2f}s)")
        if _max("recompute") >= VERDICT_THRESHOLDS["recompute_s"]:
            suspects.append(f"recompute({_max('recompute'):.2f}s)")
        tails = log_summary.get("tail_content_ms", {})
        if float(tails.get("max", 0.0)) >= VERDICT_THRESHOLDS["tail_content_ms"]:
            suspects.append(f"LLM tail({tails['max']:.0f}ms)")
    if not suspects:
        return "no suspect over threshold (synthetic gap ~0 or log without stall)"
    return "suspect: " + " / ".join(suspects)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="app log file to parse")
    ap.add_argument("--fitz-bench", action="store_true")
    ap.add_argument("--docs", type=int, default=6)
    args = ap.parse_args()

    if not args.log and not args.fitz_bench:
        args.fitz_bench = True

    log_summary: dict[str, Any] | None = None
    if args.log:
        log_summary = parse_log_file(Path(args.log))
        print(json.dumps(log_summary, indent=2, ensure_ascii=False))

    fitz: dict[str, Any] | None = None
    if args.fitz_bench:
        fitz = run_fitz_bench(args.docs)
        print(json.dumps({"fitz_bench": fitz}, indent=2, ensure_ascii=False))

    print("VERDICT:", verdict(log_summary, fitz))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"stream_gap_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "tool": "measure_stream_to_citation_gap",
                "log": log_summary,
                "fitz_bench": fitz,
                "verdict": verdict(log_summary, fitz),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
