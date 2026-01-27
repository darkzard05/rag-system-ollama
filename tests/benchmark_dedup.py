"""
Benchmark: Hash-based Deduplication vs Metadata-based Deduplication
Compares the performance of using hashlib.sha256 vs metadata tuples.
"""

import time
import hashlib
from langchain_core.documents import Document

# --- Legacy Implementation (Slow) ---
def legacy_dedup(docs):
    unique_docs = []
    seen = set()
    for doc in docs:
        doc_key = doc.page_content + doc.metadata.get("source", "")
        doc_hash = hashlib.sha256(doc_key.encode()).hexdigest()
        if doc_hash not in seen:
            unique_docs.append(doc)
            seen.add(doc_hash)
    return unique_docs

# --- Optimized Implementation (Fast) ---
def optimized_dedup(docs):
    unique_docs = []
    seen_ids = set()
    for doc in docs:
        doc_id = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_index")
        )
        if doc_id not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc_id)
    return unique_docs

def run_benchmark():
    print("🚀 Document Deduplication Benchmark")
    
    # Generate Mock Data (Many duplicates)
    NUM_TOTAL = 5000
    print(f"Generating {NUM_TOTAL} mock documents (including duplicates)...")
    docs = []
    for i in range(NUM_TOTAL):
        # Every 5th document is a duplicate
        idx = i % 1000 
        docs.append(Document(
            page_content=f"Common content block {idx} for testing purposes.",
            metadata={"source": "large_manual.pdf", "page": idx // 10, "chunk_index": idx}
        ))
        
    print("-" * 60)
    
    # Test Legacy
    start = time.perf_counter()
    res_legacy = legacy_dedup(docs)
    legacy_time = time.perf_counter() - start
    print(f"Legacy (SHA256 Hash): {legacy_time:.4f}s | Result count: {len(res_legacy)}")
    
    # Test Optimized
    start = time.perf_counter()
    res_opt = optimized_dedup(docs)
    opt_time = time.perf_counter() - start
    print(f"Optimized (Tuple ID): {opt_time:.4f}s | Result count: {len(res_opt)}")
    
    print("-" * 60)
    if legacy_time > opt_time:
        speedup = legacy_time / opt_time
        print(f"⚡ Improvement: {speedup:.2f}x faster")
    else:
        print("⚠️ No improvement (Dataset might be too small)")

if __name__ == "__main__":
    run_benchmark()
