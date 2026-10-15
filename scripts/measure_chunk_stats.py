"""Phase1 baseline chunk statistics (read-only, no pipeline changes).

Measures current chunking on tests/data/2201.07520v1.pdf:
n_chunks, token p50/p95 (HF v2-moe if available else rough),
section mixing, table preservation, REFERENCES separation,
overflow pages, t_chunk_ms, 6-query keyword coverage.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

FIXED_QUERIES: list[tuple[str, list[str]]] = [
    ("Q1 causally masked objective", ["causally masked", "masked spans"]),
    ("Q2 Table FID result", ["FID", "Table"]),
    ("Q3 Figure 1 objectives", ["Figure 1", "causal"]),
    ("Q4 size hints removal", ["size hint"]),
    ("Q5 entity disambiguation", ["entity disambiguation", "AIDA"]),
    ("Q6 training batch/GPU", ["batch", "GPU"]),
]

_MD_HDR = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)


def count_tokens(texts: list[str]) -> tuple[list[int], str]:
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
        )
        ids = tok(texts, add_special_tokens=False)["input_ids"]
        return [len(x) for x in ids], "hf-v2-moe"
    except Exception as e:
        from common.text_utils import count_tokens_rough

        print(f"[measure] HF tokenizer unavailable ({e}) -> rough fallback")
        return [count_tokens_rough(t) for t in texts], "rough"


def pct(vals: list[int], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(q * len(s)))
    return float(s[idx])


async def main() -> None:
    from core.chunking import split_documents
    from core.document_processor import load_pdf_docs

    pdf = "tests/data/2201.07520v1.pdf"
    docs = await load_pdf_docs(pdf, Path(pdf).name)
    overflow_thr = 500 * 1.5
    n_overflow = sum(1 for d in docs if len(d.page_content) > overflow_thr)

    t0 = time.perf_counter()
    try:
        from core.model_loader import load_embedding_model

        embedder = load_embedding_model()
        mode = "semantic+embedder"
    except Exception as e:
        print(
            f"[measure] embedder unavailable ({e}) -> embedder=None (recursive baseline)"
        )
        embedder = None
        mode = "recursive-baseline"
    split_docs, _ = await split_documents(docs, embedder=embedder)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    texts = [d.page_content for d in split_docs]
    toks, tok_mode = count_tokens(texts)
    mixed = sum(1 for t in texts if len(_MD_HDR.findall(t)) > 1)
    table_chunks = sum(1 for t in texts if "|" in t and "Table" in t)
    ref_chunks = sum(1 for d in split_docs if d.metadata.get("is_reference"))
    noise_as_ref = sum(1 for d in split_docs if not d.metadata.get("is_content"))

    coverage: dict[str, bool] = {}
    lowered = [t.lower() for t in texts]
    for qname, kws in FIXED_QUERIES:
        hit = any(all(k.lower() in c for k in kws) for c in lowered)
        coverage[qname] = hit

    out = {
        "pdf": pdf,
        "pages": len(docs),
        "overflow_pages": n_overflow,
        "mode": mode,
        "n_chunks": len(split_docs),
        "tok_mode": tok_mode,
        "tok_p50": pct(toks, 0.5),
        "tok_p95": pct(toks, 0.95),
        "tok_max": max(toks) if toks else 0,
        "section_mixed_chunks": mixed,
        "section_mix_rate": (mixed / len(split_docs)) if split_docs else 0.0,
        "table_chunks": table_chunks,
        "ref_chunks": ref_chunks,
        "non_content_chunks": noise_as_ref,
        "t_chunk_ms": round(dt_ms, 1),
        "query_coverage": coverage,
        "query_cover_rate": sum(coverage.values()) / len(coverage),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    Path("scripts/reports").mkdir(parents=True, exist_ok=True)
    with open("scripts/reports/chunk_baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[measure] saved scripts/reports/chunk_baseline.json")


if __name__ == "__main__":
    asyncio.run(main())
