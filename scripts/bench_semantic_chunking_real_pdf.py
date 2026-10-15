"""Real PDF Semantic Chunking Benchmark: Builtin vs Chonkie using tests/data/2201.07520v1.pdf.

Run: python scripts/bench_semantic_chunking_real_pdf.py
"""

from __future__ import annotations

import sys
import time
import asyncio
from pathlib import Path

# Ensure src is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


async def main() -> int:
    from core.document_processor import load_pdf_docs
    from core.semantic_chunker import EmbeddingBasedSemanticChunker
    from core.model_loader import ModelManager
    from core.chonkie_adapter import is_chonkie_available, chunk_with_chonkie

    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    print("=" * 70)
    print("Real PDF Semantic Chunking Benchmark (2201.07520v1.pdf)")
    print("=" * 70)

    # 1. Load PDF docs
    print("\n[1] Loading PDF documents...")
    t0 = time.perf_counter()
    raw_docs = await load_pdf_docs(pdf_path, "2201.07520v1.pdf")
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"Loaded {len(raw_docs)} pages in {t_load:.2f} ms")

    if not raw_docs:
        print("Error: No documents loaded from PDF.")
        return 1

    # Get embedder
    try:
        embedder = await ModelManager.get_embedder()
    except Exception as e:
        print(
            f"Warning: Failed to load embedder ({e}). Builtin semantic chunking may fail."
        )
        embedder = None

    # 2. Test Builtin EmbeddingBasedSemanticChunker
    print("\n[2] Testing Builtin EmbeddingBasedSemanticChunker...")
    if embedder is not None:
        try:
            t0 = time.perf_counter()
            chunker = EmbeddingBasedSemanticChunker(
                embedder=embedder,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_value=95.0,
                min_chunk_size=100,
                max_chunk_size=800,
            )
            builtin_chunks, builtin_vectors = await chunker.split_documents(raw_docs)
            t_builtin = (time.perf_counter() - t0) * 1000.0
            print(f"Builtin Semantic Chunker Results:")
            print(f"  - Time taken: {t_builtin:.2f} ms ({t_builtin / 1000:.2f} s)")
            print(f"  - Chunks generated: {len(builtin_chunks)}")
            print(
                f"  - Vectors shape: {None if builtin_vectors is None else builtin_vectors.shape}"
            )
        except Exception as e:
            print(f"Builtin semantic chunking failed: {e}")
            t_builtin = None
    else:
        print("Skipped Builtin Semantic Chunker due to missing embedder.")
        t_builtin = None

    # 3. Test Chonkie Semantic Chunker
    print("\n[3] Testing Chonkie Semantic Chunker...")
    if not is_chonkie_available():
        print(
            "Chonkie is not installed. Install with: pip install -r requirements/optional-chonkie.txt"
        )
    else:
        try:
            t0 = time.perf_counter()
            chonkie_chunks = chunk_with_chonkie(raw_docs, chunk_size=500)
            t_chonkie = (time.perf_counter() - t0) * 1000.0
            print(f"Chonkie Semantic Chunker Results:")
            print(f"  - Time taken: {t_chonkie:.2f} ms ({t_chonkie / 1000:.2f} s)")
            print(f"  - Chunks generated: {len(chonkie_chunks)}")
            # Sample sections
            sections_found = set(
                ch.metadata.get("current_section")
                for ch in chonkie_chunks
                if ch.metadata.get("current_section")
            )
            print(f"  - Distinct sections detected: {len(sections_found)}")
        except Exception as e:
            print(f"Chonkie execution failed: {e}")

    print("\n" + "=" * 70)
    print("Real PDF Benchmark completed successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
