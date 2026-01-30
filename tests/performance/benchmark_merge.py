"""
Benchmark: String Concatenation vs List Join for Document Merging
Compares the performance of the legacy merge implementation vs the optimized one.
"""

import time

from langchain_core.documents import Document


# --- Legacy Implementation (Slow) ---
def legacy_merge(docs):
    if not docs:
        return []
    sorted_docs = sorted(
        docs,
        key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("page", 0),
            d.metadata.get("chunk_index", -1),
        ),
    )
    merged = []
    current_doc = Document(
        page_content=sorted_docs[0].page_content,
        metadata=sorted_docs[0].metadata.copy(),
    )

    for next_doc in sorted_docs[1:]:
        curr_src = current_doc.metadata.get("source")
        next_src = next_doc.metadata.get("source")
        curr_idx = current_doc.metadata.get("chunk_index", -1)
        next_idx = next_doc.metadata.get("chunk_index", -1)

        if curr_src == next_src and next_idx == curr_idx + 1:
            current_doc.page_content += (
                " " + next_doc.page_content
            )  # <--- Costly operation
            current_doc.metadata["chunk_index"] = next_idx
        else:
            merged.append(current_doc)
            current_doc = Document(
                page_content=next_doc.page_content, metadata=next_doc.metadata.copy()
            )
    merged.append(current_doc)
    return merged


# --- Optimized Implementation (Fast) ---
def optimized_merge(docs):
    if not docs:
        return []
    sorted_docs = sorted(
        docs,
        key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("page", 0),
            d.metadata.get("chunk_index", -1),
        ),
    )
    merged = []

    current_meta = sorted_docs[0].metadata.copy()
    current_parts = [sorted_docs[0].page_content]

    for next_doc in sorted_docs[1:]:
        curr_src = current_meta.get("source")
        next_src = next_doc.metadata.get("source")
        curr_idx = current_meta.get("chunk_index", -1)
        next_idx = next_doc.metadata.get("chunk_index", -1)

        if curr_src == next_src and next_idx == curr_idx + 1:
            current_parts.append(next_doc.page_content)
            current_meta["chunk_index"] = next_idx
        else:
            merged.append(
                Document(page_content=" ".join(current_parts), metadata=current_meta)
            )
            current_meta = next_doc.metadata.copy()
            current_parts = [next_doc.page_content]

    merged.append(Document(page_content=" ".join(current_parts), metadata=current_meta))
    return merged


def run_benchmark():
    print("🚀 Document Merge Benchmark")

    # Generate Mock Data (Many consecutive chunks)
    NUM_CHUNKS = 10000
    print(f"Generating {NUM_CHUNKS} mock chunks...")
    docs = []
    for i in range(NUM_CHUNKS):
        docs.append(
            Document(
                page_content=f"This is a sentence part {i}. ",
                metadata={"source": "doc1.pdf", "page": 1, "chunk_index": i},
            )
        )

    print("-" * 60)

    # Test Legacy
    start = time.perf_counter()
    legacy_merge(docs)
    legacy_time = time.perf_counter() - start
    print(f"Legacy (String Concat): {legacy_time:.4f}s")

    # Test Optimized
    start = time.perf_counter()
    optimized_merge(docs)
    opt_time = time.perf_counter() - start
    print(f"Optimized (List Join):  {opt_time:.4f}s")

    print("-" * 60)
    if legacy_time > opt_time:
        speedup = legacy_time / opt_time
        print(f"⚡ Improvement: {speedup:.2f}x faster")
    else:
        print("⚠️ No improvement (Dataset might be too small)")


if __name__ == "__main__":
    run_benchmark()
