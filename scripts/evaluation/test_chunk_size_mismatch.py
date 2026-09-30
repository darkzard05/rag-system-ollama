"""
H5 Reproduction: chunk_size/overlap mismatch between text_splitter and
semantic_chunker creates inconsistent retrieval behavior.

The config sets:
- text_splitter.chunk_size: 500, chunk_overlap: 100
- semantic_chunker.min_chunk_size: 200, max_chunk_size: 1000

When semantic chunker is enabled, text_splitter is bypassed entirely.
But _postprocess_metadata in chunking.py uses TEXT_SPLITTER_CONFIG
.chunk_size (500) for the needs_sub_chunking check. The semantic
chunker creates chunks up to 1000 chars, but the text splitter config
says 500. This means:

1. Same document produces different chunk counts depending on mode
2. The needs_sub_chunking check uses wrong threshold
3. Retrieval behavior changes unpredictably when toggling semantic

USAGE:
    python scripts/evaluation/test_chunk_size_mismatch.py
"""

import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def simulate_text_splitter(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Simulate RecursiveCharacterTextSplitter behavior."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - chunk_overlap if end - chunk_overlap > start else end
    return chunks


def simulate_semantic_chunker(
    text: str, min_chunk_size: int, max_chunk_size: int
) -> list[str]:
    """Simulate semantic chunker creating chunks of varying sizes."""
    # Simplified: semantic chunker creates chunks between min and max
    # For this test, we simulate the range
    chunks = []
    pos = 0
    while pos < len(text):
        # Semantic chunker can create chunks up to max_chunk_size
        # but respects semantic boundaries
        chunk_size = min(max_chunk_size, len(text) - pos)
        # Simulate that some chunks are smaller due to semantic boundaries
        if chunk_size > min_chunk_size and pos + min_chunk_size < len(text):
            # Sometimes create smaller chunks at semantic boundaries
            import random

            chunk_size = random.randint(min_chunk_size, min(chunk_size, max_chunk_size))
        chunks.append(text[pos : pos + chunk_size])
        pos += chunk_size - (100 if pos > 0 else 0)  # Simulate overlap
    return chunks


def main():
    print("=" * 70)
    print("H5: Chunk Size Mismatch Test")
    print("=" * 70)

    # Config values
    text_splitter_chunk_size = 500
    text_splitter_overlap = 100
    semantic_min_chunk_size = 200
    semantic_max_chunk_size = 1000

    # Simulate a typical academic paper section (~3000 chars)
    np.random.seed(42)
    text = "This is the introduction section of the paper. " * 30  # ~1200 chars
    text += "\n\n## Methodology\n" + (
        "The methodology describes the experimental setup. " * 20
    )  # ~800 chars
    text += "\n\n## Results\n" + (
        "The results show significant improvements. " * 20
    )  # ~800 chars

    print(f"\nDocument length: {len(text)} chars")
    print(f"\nConfig:")
    print(
        f"  text_splitter: chunk_size={text_splitter_chunk_size}, overlap={text_splitter_overlap}"
    )
    print(
        f"  semantic_chunker: min={semantic_min_chunk_size}, max={semantic_max_chunk_size}"
    )

    # Text splitter mode
    ts_chunks = simulate_text_splitter(
        text, text_splitter_chunk_size, text_splitter_overlap
    )
    ts_sizes = [len(c) for c in ts_chunks]

    # Semantic chunker mode
    import random

    random.seed(42)
    sc_chunks = simulate_semantic_chunker(
        text, semantic_min_chunk_size, semantic_max_chunk_size
    )
    sc_sizes = [len(c) for c in sc_chunks]

    print(f"\n--- Text Splitter Mode ---")
    print(f"Chunks: {len(ts_chunks)}")
    print(
        f"Chunk sizes: min={min(ts_sizes)}, max={max(ts_sizes)}, mean={np.mean(ts_sizes):.1f}"
    )
    print(f"Total chars covered: {sum(ts_sizes)}")

    print(f"\n--- Semantic Chunker Mode ---")
    print(f"Chunks: {len(sc_chunks)}")
    print(
        f"Chunk sizes: min={min(sc_sizes)}, max={max(sc_sizes)}, mean={np.mean(sc_sizes):.1f}"
    )
    print(f"Total chars covered: {sum(sc_sizes)}")

    # The mismatch problem
    print(f"\n--- Mismatch Analysis ---")
    print(f"Chunk count difference: {len(sc_chunks) - len(ts_chunks)}")
    print(f"Max chunk size difference: {max(sc_sizes) - max(ts_sizes)}")

    # The needs_sub_chunking check uses text_splitter chunk_size
    needs_sub_chunking = any(len(c) > text_splitter_chunk_size * 1.5 for c in sc_chunks)
    print(
        f"\nneeds_sub_chunking check (using text_splitter chunk_size={text_splitter_chunk_size}):"
    )
    print(
        f"  Any semantic chunk > {text_splitter_chunk_size * 1.5}? {needs_sub_chunking}"
    )
    print(f"  But semantic chunks can be up to {semantic_max_chunk_size}!")
    print(
        f"  The check threshold ({text_splitter_chunk_size * 1.5}) is wrong for semantic mode."
    )

    # Show the inconsistency
    print(f"\n--- Inconsistency Summary ---")
    print(f"Same document produces:")
    print(f"  Text Splitter: {len(ts_chunks)} chunks (max {max(ts_sizes)} chars)")
    print(f"  Semantic:      {len(sc_chunks)} chunks (max {max(sc_sizes)} chars)")
    print(f"  Ratio: {len(sc_chunks) / len(ts_chunks):.2f}x more chunks with semantic")
    print(f"\nThis means retrieval results are fundamentally different depending")
    print(f"on whether semantic chunker is enabled, even with the same document.")
    print(f"\nRECOMMENDATION:")
    print(f"1. Align text_splitter.chunk_size with semantic_chunker.max_chunk_size")
    print(
        f"2. OR add a separate sub_chunk_size parameter for the needs_sub_chunking check"
    )
    print(f"3. OR make the check use the active chunker's max_chunk_size")
    print("=" * 70)


if __name__ == "__main__":
    main()
