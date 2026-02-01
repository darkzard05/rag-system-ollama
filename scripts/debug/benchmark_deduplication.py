
import time
import hashlib
from dataclasses import dataclass
from typing import Any

@dataclass
class MockDoc:
    page_content: str
    metadata: dict[str, Any]

def run_dedup_benchmark():
    # 1. 테스트 데이터 준비 (1,000개의 문서 조각, 50% 중복)
    base_docs = [
        MockDoc(
            page_content=f"This is a long piece of text representing document content number {i}. " * 10,
            metadata={"source": f"doc_{i//10}.pdf", "page": i % 10}
        ) for i in range(500)
    ]
    # 중복 데이터 생성
    all_documents = base_docs + base_docs
    print(f"Total documents to deduplicate: {len(all_documents)}")

    # --- Legacy: SHA256 ---
    start = time.time()
    legacy_unique = []
    seen_legacy = set()
    for doc in all_documents:
        doc_key = doc.page_content + doc.metadata.get("source", "")
        doc_hash = hashlib.sha256(doc_key.encode()).hexdigest()
        if doc_hash not in seen_legacy:
            legacy_unique.append(doc)
            seen_legacy.add(doc_hash)
    legacy_time = time.time() - start

    # --- Optimized: Tuple ---
    start = time.time()
    opt_unique = []
    seen_opt = set()
    for doc in all_documents:
        doc_id = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))
        if doc_id not in seen_opt:
            opt_unique.append(doc)
            seen_opt.add(doc_id)
    opt_time = time.time() - start

    # 결과 출력
    print(f"\n[Deduplication Benchmark Results]")
    print(f"Legacy (SHA256): {legacy_time:.6f}s")
    print(f"Optimized (Tuple): {opt_time:.6f}s")
    print(f"Speedup:          {legacy_time / opt_time:.2f}x")

    # 무결성 검사
    match = len(legacy_unique) == len(opt_unique)
    if match:
        for d1, d2 in zip(legacy_unique, opt_unique):
            if d1.page_content != d2.page_content or d1.metadata != d2.metadata:
                match = False
                break
    print(f"\nIntegrity Check: {'PASSED' if match else 'FAILED'}")

if __name__ == "__main__":
    run_dedup_benchmark()
