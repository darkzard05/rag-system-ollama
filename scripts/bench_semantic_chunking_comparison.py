"""Semantic chunking benchmark: Builtin EmbeddingBasedSemanticChunker vs Chonkie SemanticChunker.

Run: python scripts/bench_semantic_chunking_comparison.py
"""

from __future__ import annotations

import sys
import time
import asyncio
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SAMPLE_LARGE = """# Abstract
This document evaluates semantic chunking performance in retrieval-augmented generation systems.
Local LLMs and embedding models introduce distinct latency profiles compared to cloud APIs.

# Introduction
Retrieval-augmented generation (RAG) relies heavily on accurate document splitting.
Traditional fixed-size chunking often splits sentences and destroys topical coherence.
Semantic chunking addresses this by computing embedding similarity across consecutive sentences.
However, computing embeddings for every sentence introduces significant computational overhead.

# Methodology
We benchmark two semantic chunking approaches:
1. Builtin EmbeddingBasedSemanticChunker with sentence-level similarity and L2 normalization.
2. Chonkie SemanticChunker with header-aware pre-splitting and optimized token counting.

# Experiments & Results
Experiments are conducted on multi-page documents containing markdown headers, lists, and tables.
| Strategy | Latency (ms) | Chunk Count | Topical Coherence |
|---|---|---|---|
| Builtin | 1250.4 | 12 | High |
| Chonkie | 320.1 | 10 | High |

# Conclusion
Header-aware pre-splitting combined with high-performance semantic engines reduces indexing latency by over 70% while preserving structural context.
"""


async def main() -> int:
    from langchain_core.documents import Document
    from core.semantic_chunker import EmbeddingBasedSemanticChunker
    from core.model_loader import ModelManager
    from core.chonkie_adapter import is_chonkie_available, chunk_with_chonkie

    docs = [Document(page_content=SAMPLE_LARGE, metadata={"source": "benchmark_pdf"})]

    print("=" * 60)
    print("Semantic Chunking Performance Benchmark")
    print("=" * 60)
    print(f"Sample text length: {len(SAMPLE_LARGE)} characters")

    # 1. Test Builtin Semantic Chunker (without embedder or with dummy embedder/mock if needed, or real model if available)
    print("\n--- [1] Testing Builtin EmbeddingBasedSemanticChunker ---")
    try:
        embedder = await ModelManager.get_embedder()
        t0 = time.perf_counter()
        chunker = EmbeddingBasedSemanticChunker(
            embedder=embedder,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_value=95.0,
            min_chunk_size=50,
            max_chunk_size=500,
        )
        builtin_chunks, builtin_vectors = await chunker.split_documents(docs)
        t_builtin = (time.perf_counter() - t0) * 1000.0
        print(f"Builtin Semantic Chunker Result:")
        print(f"  - Time taken: {t_builtin:.2f} ms")
        print(f"  - Chunks generated: {len(builtin_chunks)}")
        print(
            f"  - Vectors shape: {None if builtin_vectors is None else builtin_vectors.shape}"
        )
    except Exception as e:
        print(f"Builtin Semantic Chunker skipped or failed (e.g. Ollama offline): {e}")
        t_builtin = None

    # 2. Test Chonkie Semantic Chunker (if available)
    print("\n--- [2] Testing Chonkie Semantic Chunker ---")
    if not is_chonkie_available():
        print("Chonkie is not installed in the current environment.")
        print("To install: pip install -r requirements/optional-chonkie.txt")
    else:
        try:
            t0 = time.perf_counter()
            chonkie_chunks = chunk_with_chonkie(docs, chunk_size=300)
            t_chonkie = (time.perf_counter() - t0) * 1000.0
            print(f"Chonkie Semantic Chunker Result:")
            print(f"  - Time taken: {t_chonkie:.2f} ms")
            print(f"  - Chunks generated: {len(chonkie_chunks)}")
            for i, ch in enumerate(chonkie_chunks[:3]):
                print(
                    f"    [{i + 1}] section={ch.metadata.get('current_section')!r} length={len(ch.page_content)}"
                )
        except Exception as e:
            print(f"Chonkie execution failed: {e}")

    print("\n" + "=" * 60)
    print("Benchmark completed successfully.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
