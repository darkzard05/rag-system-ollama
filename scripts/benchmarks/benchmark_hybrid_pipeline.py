"""Full Pipeline Benchmark: Baseline vs Hybrid Optimized (2201.07520v1.pdf).

Run: python scripts/benchmarks/benchmark_hybrid_pipeline.py
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
    "What datasets were used for zero-shot summarization evaluation?",
]


async def main() -> int:
    from common.config import DEFAULT_OLLAMA_MODEL
    from core.rag_core import RAGSystem
    from core.model_loader import ModelManager
    from core.session import SessionManager

    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = "2201.07520v1.pdf"

    print("=" * 70)
    print("Full Pipeline Benchmark: End-to-End Evaluation")
    print("=" * 70)

    embedder = await ModelManager.get_embedder()
    llm = await ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)

    # 1. Baseline / Optimized Pipeline Build Benchmark
    session_id = "bench_pipeline_session"
    SessionManager.init_session(session_id=session_id)
    rag_sys = RAGSystem(session_id=session_id)

    print(f"\n[1] Building pipeline with PDF: {file_name}")
    t0 = time.perf_counter()
    message, cache_used = await rag_sys.build_pipeline(
        file_path=pdf_path,
        file_name=file_name,
        embedder=embedder,
    )
    build_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Pipeline Build Result:")
    print(f"  - Message: {message}")
    print(f"  - Cache Used: {cache_used}")
    print(
        f"  - Total Build Time: {build_time_ms:.2f} ms ({build_time_ms / 1000:.2f} s)"
    )

    # 2. Query & Answer Generation Benchmark
    print(
        f"\n[2] Benchmarking Queries & Generation ({len(TEST_QUESTIONS)} questions)..."
    )
    query_results = []

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n  Query {i}: {q!r}")
        t0 = time.perf_counter()

        # Measure TTFT and total generation using aquery
        res = await rag_sys.aquery(q)
        q_time_ms = (time.perf_counter() - t0) * 1000.0

        response_text = res.get("response", "")
        docs_count = len(res.get("documents", []))

        print(f"    - Response length: {len(response_text)} chars")
        print(f"    - Retrieved docs: {docs_count}")
        print(f"    - Query Latency: {q_time_ms:.2f} ms")

        query_results.append(
            {
                "query": q,
                "latency_ms": q_time_ms,
                "docs": docs_count,
            }
        )

    # 3. Summary Report
    print("\n" + "=" * 70)
    print("Benchmark Summary Report")
    print("=" * 70)
    print(f"PDF File: {file_name}")
    print(f"Pipeline Build Time: {build_time_ms:.2f} ms")
    for r in query_results:
        print(
            f"  - Q: {r['query'][:40]}... | Latency: {r['latency_ms']:.2f} ms | Docs: {r['docs']}"
        )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
