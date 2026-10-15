"""Full Pipeline Comparative Benchmark: Legacy vs Hybrid Optimized (Answers & Latency).

Target: tests/data/2201.07520v1.pdf
Run: python scripts/benchmarks/compare_full_pipeline.py
"""

from __future__ import annotations

import sys
import time
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

TEST_QUESTIONS = [
    "What is the core architecture of CM3?",
    "How does CM3 compare to DALL-E in image generation?",
]


async def run_pipeline_test(
    pipeline_name: str, session_id: str, use_semantic: bool
) -> None:
    from core.rag_core import RAGSystem
    from core.model_loader import ModelManager
    from core.session import SessionManager
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import core.chunking as chunking_module

    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = "2201.07520v1.pdf"

    print("=" * 80)
    print(f"Pipeline Test: {pipeline_name}")
    print("=" * 80)

    embedder = await ModelManager.get_embedder()
    SessionManager.init_session(session_id=session_id)
    rag_sys = RAGSystem(session_id=session_id)

    # If legacy (use_semantic=False), we monkeypatch split_documents or use recursive splitter
    original_split_docs = chunking_module.split_documents
    if not use_semantic:

        async def legacy_split_documents(docs, embedder=None, session_id=None):
            from common.config import TEXT_SPLITTER_CONFIG
            from langchain_core.documents import Document
            import numpy as np

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG.get("chunk_size", 500),
                chunk_overlap=TEXT_SPLITTER_CONFIG.get("chunk_overlap", 100),
            )
            split_docs = await asyncio.to_thread(splitter.split_documents, docs)
            vectors = None
            if embedder:
                vectors = await chunking_module._embed_documents_chunks(
                    split_docs, embedder
                )
            return split_docs, vectors

        chunking_module.split_documents = legacy_split_documents

    try:
        t0 = time.perf_counter()
        message, cache_used = await rag_sys.build_pipeline(
            file_path=pdf_path,
            file_name=file_name,
            embedder=embedder,
        )
        build_time_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[{pipeline_name}] Build Time: {build_time_ms:.2f} ms | Cache: {cache_used}"
        )

        for idx, q in enumerate(TEST_QUESTIONS, 1):
            print(f"\n  --- Query {idx}: {q!r} ---")
            t0 = time.perf_counter()
            res = await rag_sys.aquery(q)
            q_time_ms = (time.perf_counter() - t0) * 1000.0

            response = res.get("response", "")
            docs = res.get("documents", [])
            print(f"  Query Latency: {q_time_ms:.2f} ms | Retrieved Docs: {len(docs)}")
            print(
                f"  Generated Answer:\n{response[:600]}...\n"
                if len(response) > 600
                else f"  Generated Answer:\n{response}\n"
            )

    finally:
        chunking_module.split_documents = original_split_docs


async def main() -> int:
    print("Starting Full Pipeline Comparison (Legacy vs Hybrid Optimized)...")

    # 1. Legacy Pipeline
    await run_pipeline_test(
        "Legacy Recursive Pipeline", "session_legacy", use_semantic=False
    )

    # 2. Hybrid Optimized Pipeline
    await run_pipeline_test(
        "Hybrid Optimized Pipeline", "session_hybrid", use_semantic=True
    )

    print("\n" + "=" * 80)
    print("Full Pipeline Comparative Benchmark Completed.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
