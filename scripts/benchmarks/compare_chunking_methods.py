"""Comparative Benchmark: Legacy Semantic Chunking vs Hybrid Optimized Chunking.

Target: tests/data/2201.07520v1.pdf
Run: python scripts/benchmarks/compare_chunking_methods.py
"""

from __future__ import annotations

import sys
import time
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


async def main() -> int:
    from core.document_processor import load_pdf_docs
    from core.model_loader import ModelManager
    from core.semantic_chunker import EmbeddingBasedSemanticChunker
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from common.config import SEMANTIC_CHUNKER_CONFIG

    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    print("=" * 70)
    print("Comparative Benchmark: Legacy vs Hybrid Optimized Chunking")
    print("=" * 70)

    # 1. Load PDF docs
    print("\n[1] Loading PDF documents...")
    t0 = time.perf_counter()
    raw_docs = await load_pdf_docs(pdf_path, "2201.07520v1.pdf")
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"Loaded {len(raw_docs)} pages in {t_load:.2f} ms")

    embedder = await ModelManager.get_embedder()

    # 2. Benchmark Method A: Legacy Recursive / Standard Splitter (Baseline)
    print("\n[2] Benchmarking Legacy Recursive Splitter (Baseline)...")
    t0 = time.perf_counter()
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    legacy_chunks = recursive_splitter.split_documents(raw_docs)
    t_legacy = (time.perf_counter() - t0) * 1000.0
    print(f"Legacy Splitter Results:")
    print(f"  - Time taken: {t_legacy:.2f} ms")
    print(f"  - Chunks generated: {len(legacy_chunks)}")

    # 3. Benchmark Method B: Hybrid Optimized Semantic Chunker (Current Optimized)
    print("\n[3] Benchmarking Hybrid Optimized Semantic Chunker...")
    t0 = time.perf_counter()

    # Temporarily enable semantic chunking config if disabled
    from core.chunking import split_documents

    optimized_chunks, optimized_vectors = await split_documents(
        raw_docs, embedder=embedder
    )
    t_hybrid = (time.perf_counter() - t0) * 1000.0

    print(f"Hybrid Optimized Chunker Results:")
    print(f"  - Time taken: {t_hybrid:.2f} ms ({t_hybrid / 1000:.2f} s)")
    print(f"  - Chunks generated: {len(optimized_chunks)}")
    print(
        f"  - Vectors count: {None if optimized_vectors is None else len(optimized_vectors)}"
    )

    # 4. Summary Comparison Table
    print("\n" + "=" * 70)
    print("Comparison Summary Report")
    print("=" * 70)
    print(f"{'Method':<35} | {'Time (ms)':<12} | {'Chunks':<8}")
    print("-" * 70)
    print(
        f"{'Legacy Recursive Splitter':<35} | {t_legacy:<12.2f} | {len(legacy_chunks):<8}"
    )
    print(
        f"{'Hybrid Optimized Chunker':<35} | {t_hybrid:<12.2f} | {len(optimized_chunks):<8}"
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
