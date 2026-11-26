"""Reproduce reference-section citations for short keyword queries.

Builds the CM3 paper exactly as the app does, runs the given queries through
``RAGSystem.astream``, and prints every retrieved chunk with its page/section
plus the final citation payload. Use it to check which document part a
citation points at.

Usage:
    python scripts/evaluation/repro_reference_citations.py
    python scripts/evaluation/repro_reference_citations.py --query vqvae-gan
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

PDF_PATH = ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf"

from common.utils import doc_stable_id  # noqa: E402


def doc_summary(doc: Any) -> dict[str, Any]:
    content = getattr(doc, "page_content", "") or ""
    metadata = getattr(doc, "metadata", {}) or {}
    section = metadata.get("section_path") or metadata.get("current_section")
    return {
        "doc_id": doc_stable_id(doc),
        "page": metadata.get("page"),
        "section": section,
        "is_reference": bool(metadata.get("is_reference")),
        "rerank_score": metadata.get("rerank_score"),
        "head": content[:220].replace("\n", " "),
    }


def is_reference_like(summary: dict[str, Any]) -> bool:
    section = str(summary.get("section") or "").lower()
    return bool(summary.get("is_reference")) or bool(
        re.search(r"\b(references?|bibliography)\b", section)
    )


async def run_query(rag: Any, query: str, model: str) -> int:
    docs: list[Any] = []
    citations: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    stream = await rag.astream(query, model_name=model)
    try:
        async for kind, data in stream:
            if not isinstance(data, dict):
                continue
            if kind == "updates":
                retrieved = data.get("retrieve") or {}
                if isinstance(retrieved, dict):
                    batch = retrieved.get("relevant_docs") or []
                    if batch:
                        docs = list(batch)
            elif kind == "custom":
                content = data.get("content")
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                payload = data.get("citations")
                if isinstance(payload, list):
                    citations = payload
            elif kind == "replace":
                answer_parts = [str(data.get("content") or "")]
                payload = data.get("citations")
                if isinstance(payload, list):
                    citations = payload
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()

    summaries = [doc_summary(doc) for doc in docs]
    suspicious = [item for item in summaries if is_reference_like(item)]
    suspicious_ids = {item["doc_id"] for item in suspicious}
    cited_ids = {str(cit.get("doc_id")) for cit in citations}
    print(f"\n=== query: {query!r} ===")
    print(f"retrieved={len(summaries)} citations={len(citations)}")
    for index, item in enumerate(summaries, start=1):
        marker = " <== REFERENCE-LIKE" if is_reference_like(item) else ""
        print(
            f"  [{index:02d}] id={item['doc_id']} page={item['page']} "
            f"section={item['section']!r} is_ref={item['is_reference']} "
            f"rerank={item['rerank_score']}{marker}"
        )
        print(f"       {item['head']}")
    print(f"answer_head={''.join(answer_parts)[:240]!r}")
    print(f"citations={json.dumps(citations, ensure_ascii=False)[:1200]}")
    print(
        f"reference_like={len(suspicious)} cited_reference_like={len(suspicious_ids & cited_ids)}"
    )
    return len(suspicious)


async def main(queries: list[str]) -> int:
    import common.config as config_module
    from common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL
    from common.logging_config import setup_logging
    from core.graph import _generate, _graph_core, _retrieval
    from core.model_loader import ModelManager
    from core.rag_core import RAGSystem
    from core.session import SessionManager

    # A cached answer would skip retrieval, so keep the repro deterministic.
    config_module.QUERY_CACHE_ENABLED = False
    for module in (_generate, _graph_core, _retrieval):
        if hasattr(module, "QUERY_CACHE_ENABLED"):
            module.QUERY_CACHE_ENABLED = False

    setup_logging(log_level="WARNING", log_file=None, console_output=False)
    session_id = f"repro-ref-cites-{int(time.time())}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfile(PDF_PATH, tmp.name)
        temp_pdf = tmp.name

    try:
        embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
        message, cached = await rag.build_pipeline(temp_pdf, PDF_PATH.name, embedder)
        print(f"build: {message!r} cache_used={cached}")
        total = 0
        for query in queries:
            total += await run_query(rag, query, DEFAULT_OLLAMA_MODEL)
        print(f"\nREFERENCE_LIKE_CHUNKS={total}")
        return 0
    finally:
        rag.clear_session()
        SessionManager.delete_session(session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="query to run (repeatable); defaults to cm3 and vqvae-gan",
    )
    args = parser.parse_args()
    asyncio.run(main(args.queries or ["cm3", "vqvae-gan"]))
