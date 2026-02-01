import time
import os
import numpy as np
from langchain_core.documents import Document
from src.core.semantic_chunker import EmbeddingBasedSemanticChunker

def run_fair_benchmark():
    import fitz
    pdf_path = "tests/data/2201.07520v1.pdf"
    
    # 1. 테스트 데이터 준비 (대규모: 1,000페이지 분량)
    with fitz.open(pdf_path) as doc:
        pages_content = [page.get_text() for page in doc]
    
    replications = 50 
    raw_docs = []
    for i in range(replications):
        for p_idx, content in enumerate(pages_content):
            raw_docs.append(Document(
                page_content=content, 
                metadata={"page": p_idx + 1, "rep": i}
            ))
    
    print(f"Total simulated pages: {len(raw_docs)}")
    
    # 미리 10,000개의 가상 청크 생성 (오프셋 포함)
    total_len = sum(len(d.page_content) for d in raw_docs)
    chunk_dicts = []
    for i in range(10000):
        start = (total_len // 10000) * i
        end = start + 100
        chunk_dicts.append({"text": "sample", "start": start, "end": end, "vector": None})

    # --- 1. String Concatenation Benchmark ---
    print("\n--- [1] String Concatenation Performance ---")
    
    # Legacy: +=
    start = time.time()
    full_text_legacy = ""
    for d in raw_docs:
        full_text_legacy += d.page_content + " "
    legacy_concat_time = time.time() - start
    
    # Optimized: join()
    start = time.time()
    full_text_opt = "".join([d.page_content + " " for d in raw_docs])
    opt_concat_time = time.time() - start
    
    print(f"Legacy (+=):   {legacy_concat_time:.4f}s")
    print(f"Optimized (join): {opt_concat_time:.4f}s")
    print(f"Concat Speedup: {legacy_concat_time / opt_concat_time:.2f}x")

    # --- 2. Metadata Mapping Benchmark ---
    print("\n--- [2] Metadata Mapping Performance ---")
    
    # 전처리: doc_ranges 생성
    doc_ranges = []
    curr = 0
    for d in raw_docs:
        doc_ranges.append({"start": curr, "end": curr + len(d.page_content), "metadata": d.metadata})
        curr += len(d.page_content) + 1

    # Legacy: Full Linear Search O(N*M)
    start = time.time()
    for chunk in chunk_dicts:
        c_center = (chunk["start"] + chunk["end"]) // 2
        for r in doc_ranges:
            if r["start"] <= c_center < r["end"]:
                _ = r["metadata"]
                break
    legacy_map_time = time.time() - start

    # Optimized: last_found_idx 기반 검색 O(N+M)
    start = time.time()
    last_found_idx = 0
    for chunk in chunk_dicts:
        c_center = (chunk["start"] + chunk["end"]) // 2
        for i in range(last_found_idx, len(doc_ranges)):
            r = doc_ranges[i]
            if r["start"] <= c_center < r["end"]:
                _ = r["metadata"]
                last_found_idx = i
                break
    opt_map_time = time.time() - start

    print(f"Legacy (Full Scan): {legacy_map_time:.4f}s")
    print(f"Optimized (Index):  {opt_map_time:.4f}s")
    print(f"Mapping Speedup:    {legacy_map_time / opt_map_time:.2f}x")

if __name__ == "__main__":
    run_fair_benchmark()