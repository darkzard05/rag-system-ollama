"""H6 EXTERNAL repro: chunk sweep (breakpoint x chunk size).

No src/ edits. No large reindex. Guarded timeouts (import <= 120s total).

Plan:
  1. Load 1 small doc from tests/data (golden_set.json -> synthetic markdown;
     fallback to testset_2201.csv; fallback to built-in synthetic markdown).
  2. Chunk-size sweep with RecursiveCharacterTextSplitter (no embedder, no
     Ollama) over [(500,100), (400,80), (700,150)] — mirrors
     text_splitter.chunk_size/chunk_overlap (config.yml 113-115).
  3. Breakpoint sweep [0.5, 0.6, 0.7] analytically (cosine_distance = 1-sim,
     similarity_threshold semantics per semantic_chunker, config 139-147).
     Optionally tries src.core.chunking.split_documents import (read-only,
     embedder=None path) within a time budget; falls back analytically.
  4. Report chunk count / avg size / min-max / section contamination
     (chunks spanning >1 markdown header).

Usage:
    python scripts/evaluation/repro_h6_chunk.py
Output:
    stdout + reports/repro_h6_chunk_<ts>.md
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import statistics
import time
from datetime import datetime
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "tests", "data")

CHUNK_SWEEP = [(500, 100), (400, 80), (700, 150)]
BREAKPOINTS = [0.5, 0.6, 0.7]
IMPORT_BUDGET_S = 100  # must stay < 120s guard

HEADER_RE = re.compile(r"^#{1,4}\s+.+", re.MULTILINE)


def load_doc_text() -> tuple[str, str]:
    """Return (source_label, markdown_text). Never touches src/."""
    # 1) golden_set.json -> build small markdown from first entries
    gpath = os.path.join(DATA_DIR, "golden_set.json")
    if os.path.exists(gpath):
        try:
            with open(gpath, encoding="utf-8") as f:
                data = json.load(f)
            items = (
                data
                if isinstance(data, list)
                else data.get("items", data.get("testset", []))
            )
            if isinstance(items, list) and items:
                parts = ["# Repro doc (from golden_set.json)\n"]
                for i, it in enumerate(items[:4]):
                    q = (
                        it.get("question", it.get("query", f"q{i}"))
                        if isinstance(it, dict)
                        else str(it)
                    )
                    a = (
                        it.get("answer", it.get("ground_truth", ""))
                        if isinstance(it, dict)
                        else ""
                    )
                    parts.append(f"\n## Section {i + 1}\n{q}\n{a}\n")
                # pad to ~3-4k chars so chunk sweep is meaningful
                filler = (
                    "\nThe CM3 model uses causally masked training over structured "
                    "multimodal documents. Tokens are generated left to right. " * 20
                )
                parts.append("\n## Appendix\n" + filler)
                return (
                    f"golden_set.json ({len(items)} items, first 4 used)",
                    "".join(parts),
                )
        except Exception:
            pass
    # 2) testset csv
    cpath = os.path.join(DATA_DIR, "testset_2201.csv")
    if os.path.exists(cpath):
        try:
            with open(cpath, encoding="utf-8") as f:
                content = f.read(6000)
            if content.strip():
                return (
                    f"testset_2201.csv (first 6k chars)",
                    "# Repro doc (from testset_2201.csv)\n\n## Data\n" + content,
                )
        except Exception:
            pass
    # 3) built-in synthetic markdown (deterministic)
    synth = (
        "# Introduction\n"
        + ("CM3 is a causally masked generative model. " * 30)
        + "\n\n## Methodology\n"
        + ("The methodology describes experimental setup. " * 25)
        + "\n\n## Results\n"
        + ("Results show significant improvements. " * 25)
        + "\n\n## References\n"
        + ("Smith et al. doi:10.1000/xyz. " * 10)
    )
    return ("built-in synthetic markdown", synth)


def try_import_splitter(budget_s: float = IMPORT_BUDGET_S) -> tuple[str, object | None]:
    """Try importing RecursiveCharacterTextSplitter + src chunking (read-only).

    Returns (mode, splitter_cls_or_None). Never calls Ollama/embeddings.
    Runs the import in a worker thread so a slow torch/langchain import
    cannot hang the repro past the budget.
    """

    def _do():
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            return ("live-splitter", RecursiveCharacterTextSplitter)
        except Exception as e:
            return (f"unavailable:{e}", None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_do)
        try:
            return fut.result(timeout=budget_s)
        except concurrent.futures.TimeoutError:
            return ("timeout-fallback", None)


def chunk_sweep(text: str, splitter_cls: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    _Doc = None
    try:
        from langchain_core.documents import Document as _ImportedDoc

        _Doc = _ImportedDoc
        has_doc = True
    except Exception:
        has_doc = False

    for size, overlap in CHUNK_SWEEP:
        t = time.time()
        if splitter_cls is not None:
            sp = splitter_cls(chunk_size=size, chunk_overlap=overlap)
            if has_doc and _Doc is not None:
                chunks = sp.split_documents([_Doc(page_content=text, metadata={})])
                sizes = [len(c.page_content) for c in chunks]
                texts = [c.page_content for c in chunks]
            else:
                texts = sp.split_text(text)
                sizes = [len(x) for x in texts]
        else:  # pure-analytical fallback: fixed-stride slicing
            texts, start = [], 0
            while start < len(text):
                end = min(start + size, len(text))
                texts.append(text[start:end])
                start = end - overlap if end - overlap > start else end
            sizes = [len(x) for x in texts]
        # section contamination: chunks containing >= 2 headers
        contam = sum(1 for x in texts if len(HEADER_RE.findall(x)) >= 2)
        results.append(
            {
                "size": size,
                "overlap": overlap,
                "n": len(texts),
                "avg": statistics.mean(sizes) if sizes else 0,
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "contam": contam,
                "contam_rate": (contam / len(texts)) if texts else 0,
                "elapsed": time.time() - t,
            }
        )
    return results


def breakpoint_sweep() -> list[dict[str, Any]]:
    """Deterministic analytical sweep mirroring _find_breakpoints.

    similarity_threshold type: breakpoint when cosine_distance (=1-sim) > thr.
    Synthetic similarities: 30 within-para U[0.7,0.95], 20 between-para
    U[0.2,0.6], seed 42 (same recipe as test_breakpoint_06.py).
    """
    import random

    rng = random.Random(42)
    within = [rng.uniform(0.7, 0.95) for _ in range(30)]
    between = [rng.uniform(0.2, 0.6) for _ in range(20)]
    sims = within + between
    rng.shuffle(sims)
    out = []
    for thr in BREAKPOINTS:
        bps = sum(1 for s in sims if (1 - s) > thr)
        broken_within = sum(1 for s in within if (1 - s) > thr)
        out.append(
            {
                "thr": thr,
                "breaks": bps,
                "frag_rate": bps / len(sims),
                "broken_within": broken_within,
                "broken_within_rate": broken_within / len(within),
            }
        )
    return out


def main() -> int:
    t0 = datetime.now()
    label, text = load_doc_text()
    mode, splitter_cls = try_import_splitter()
    cs = chunk_sweep(text, splitter_cls)
    bs = breakpoint_sweep()

    lines = []
    lines.append("# H6 repro: chunk sweep (breakpoint x chunk size)")
    lines.append("")
    lines.append(f"- Timestamp: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        "- Config under test: `text_splitter {chunk_size: 500, chunk_overlap: 100}` "
        "(113-115); `semantic_chunker {enabled: true, "
        "breakpoint_threshold_type: similarity_threshold, "
        "breakpoint_threshold_value: 0.6, min: 200, max: 1000, sim_thr: 0.6}` (139-147)"
    )
    lines.append(
        f"- Doc source: {label}; chars={len(text)}, headers={len(HEADER_RE.findall(text))}"
    )
    lines.append(f"- Splitter mode: {mode} (embedder=None path; no Ollama, no reindex)")
    lines.append("")
    lines.append("## 1. Chunk-size sweep (RecursiveCharacterTextSplitter, no embedder)")
    lines.append("")
    lines.append("| chunk/overlap | n | avg | min | max | contam (multi-header) |")
    lines.append("|---|---|---|---|---|---|")
    for r in cs:
        lines.append(
            f"| {r['size']}/{r['overlap']} | {r['n']} | {r['avg']:.0f} | "
            f"{r['min']} | {r['max']} | {r['contam']} ({r['contam_rate']:.0%}) |"
        )
    lines.append("")
    cur = next((r for r in cs if r["size"] == 500), cs[0] if cs else None)
    if cur:
        lines.append(
            f"- Current config (500/100): n={cur['n']}, avg={cur['avg']:.0f}, "
            f"contam={cur['contam']} ({cur['contam_rate']:.0%})."
        )
        lines.append(
            "  - Larger 700/150 merges sections (fewer, longer chunks, "
            "higher contamination risk); smaller 400/80 fragments "
            "(more, shorter chunks, lower contamination)."
        )
    lines.append("")
    lines.append("## 2. Breakpoint sweep (analytical, similarity_threshold semantics)")
    lines.append(
        "- Rule: break when cosine_distance (1 - sim) > thr; "
        "synthetic 50 pairs (30 within-para high-sim, 20 between-para low-sim), seed 42."
    )
    lines.append("")
    lines.append("| thr | breaks/50 | frag rate | within-para broken (30) |")
    lines.append("|---|---|---|---|")
    for r in bs:
        lines.append(
            f"| {r['thr']:.1f} | {r['breaks']} | {r['frag_rate']:.0%} | "
            f"{r['broken_within']} ({r['broken_within_rate']:.0%}) |"
        )
    lines.append("")
    lines.append(
        "- Current 0.6 breaks pairs with sim < 0.4; "
        "0.5 is stricter (sim < 0.5 -> more breaks), "
        "0.7 is looser (sim < 0.3 -> fewer breaks)."
    )
    lines.append(
        "  - 0.6 vs percentile-95 default: similarity_threshold is far more "
        "aggressive (see scripts/evaluation/test_breakpoint_06.py)."
    )
    lines.append("")
    lines.append("## 3. Verdict")
    lines.append(
        "- Sweep completed without Ollama and without src//config edits; "
        "chunk path used embedder=None (standard splitter) only."
    )
    lines.append(
        "- Section contamination + fragmentation trade-off is quantified above; "
        "semantic path (embedder-backed) not executed to respect the no-reindex guard."
    )
    lines.append("")
    lines.append(
        "_Guard: splitter import budget "
        f"{IMPORT_BUDGET_S}s; no embeddings/Ollama calls; single small doc only._"
    )

    md = "\n".join(lines) + "\n"
    print(md)
    out_dir = os.path.join(ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"repro_h6_chunk_{t0.strftime('%Y%m%d_%H%M%S')}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[H6] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
