"""Chunker spike: builtin RecursiveCharacter vs chonkie (if installed).

Run: python scripts/bench_chunker_parity.py
- Always runs builtin baseline (no model needed).
- chonkie branch runs only when installed; otherwise SKIP (spike).
"""

from __future__ import annotations

import sys
import time

SAMPLE = """# Introduction
Artificial intelligence retrieves context from documents. Semantic chunking
keeps related sentences together instead of cutting at fixed lengths.

# Methods
We compare recursive splitting with semantic splitting. Tables and figures
interrupt reading order. Complex layouts need structure-aware parsing.

# Results
Semantic chunks preserve topical coherence. Retrieval recall improves when
chunk boundaries respect section structure and table integrity.
"""


def main() -> int:
    sys.path.insert(0, "src")
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from core.chonkie_adapter import is_chonkie_available, split_by_headers

    docs = [Document(page_content=SAMPLE, metadata={"source": "spike"})]

    t0 = time.perf_counter()
    rec = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    builtin = rec.split_documents(docs)
    b_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[chunker] builtin recursive: {len(builtin)} chunks, {b_ms:.1f}ms")
    for d in builtin[:3]:
        print(f"  - {len(d.page_content)} chars: {d.page_content[:70]!r}")

    sections = split_by_headers(SAMPLE)
    print(
        f"[chunker] header split: {len(sections)} sections: {[s for s, _ in sections]}"
    )

    if not is_chonkie_available():
        print("[chunker] chonkie not installed -> SKIP (spike only)")
        print("[chunker] install: pip install -r requirements/optional-chonkie.txt")
        return 0

    from core.chonkie_adapter import chunk_with_chonkie

    t0 = time.perf_counter()
    out = chunk_with_chonkie(docs)
    c_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[chunker] chonkie semantic: {len(out)} chunks, {c_ms:.1f}ms")
    for d in out[:5]:
        print(f"  - [{d.metadata.get('current_section')}] {len(d.page_content)} chars")
    print("[chunker] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
