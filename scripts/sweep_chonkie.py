"""Phase4 — chonkie 12-cell sweep on CM3 paper (single-doc).

Grid: threshold {0.6,0.7,0.8} x similarity_window {1,3} x skip_window {0,1}.
Winner: coverage desc, tok_p95 asc, section_mix asc (pre-registered tie-break).
Writes scripts/reports/chonkie_sweep.json.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from scripts.measure_chunk_stats import FIXED_QUERIES  # noqa: E402

_MD_HDR = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)


def main() -> None:
    from transformers import AutoTokenizer

    from core.chonkie_adapter import chunk_with_chonkie
    from core.chunking import _postprocess_metadata
    from core.document_processor import load_pdf_docs

    tok = AutoTokenizer.from_pretrained(
        "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
    )
    docs = asyncio.run(load_pdf_docs("tests/data/2201.07520v1.pdf", "2201.07520v1.pdf"))

    rows: list[dict] = []
    grid = list(itertools.product([0.6, 0.7, 0.8], [1, 3], [0, 1]))
    for thr, win, skip in grid:
        t0 = time.perf_counter()
        out = chunk_with_chonkie(
            docs,
            threshold=thr,
            similarity_window=win,
            skip_window=skip,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _postprocess_metadata(out)
        texts = [d.page_content for d in out]
        toks = sorted(len(t) for t in tok(texts, add_special_tokens=False)["input_ids"])
        mixed = sum(1 for t in texts if len(_MD_HDR.findall(t)) > 1)
        lowered = [t.lower() for t in texts]
        cov = sum(
            1
            for _, kws in FIXED_QUERIES
            if any(all(k.lower() in c for k in kws) for c in lowered)
        )
        rows.append(
            {
                "threshold": thr,
                "similarity_window": win,
                "skip_window": skip,
                "n_chunks": len(out),
                "tok_p50": float(toks[len(toks) // 2]),
                "tok_p95": float(toks[min(len(toks) - 1, int(len(toks) * 0.95))]),
                "tok_max": float(toks[-1]),
                "section_mix": mixed,
                "table_chunks": sum(1 for t in texts if "|" in t and "Table" in t),
                "ref_chunks": sum(1 for d in out if d.metadata.get("is_reference")),
                "coverage": cov,
                "t_chunk_ms": round(dt_ms, 1),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))
    rows.sort(key=lambda r: (-r["coverage"], r["tok_p95"], r["section_mix"]))
    winner = rows[0]
    print("WINNER: " + json.dumps(winner, ensure_ascii=False))
    Path("scripts/reports").mkdir(parents=True, exist_ok=True)
    with open("scripts/reports/chonkie_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "winner": winner}, f, indent=2)
    print("[sweep] saved scripts/reports/chonkie_sweep.json")


if __name__ == "__main__":
    main()
