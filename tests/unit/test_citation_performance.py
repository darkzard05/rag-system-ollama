# tests/unit/test_citation_performance.py
import time
from src.common.utils import apply_tooltips_to_response
from langchain_core.documents import Document


def test_citation_matching_performance():
    # 1. 10000개의 대규모 문서 청크 생성 (M=10000)
    docs = [
        Document(
            page_content=f"Content of section {i} on page {i // 10 + 1}",
            metadata={"page": i // 10 + 1, "current_section": f"Section {i}"},
        )
        for i in range(10000)
    ]

    # 2. 100개의 인용이 포함된 답변 생성 (N=100)
    response_text = "Analysis shows " + " ".join(
        [f"fact {i} [Section {i * 100}, p.{i * 10 + 1}]" for i in range(100)]
    )

    # 3. 성능 측정
    start = time.time()
    processed_text = apply_tooltips_to_response(response_text, docs)
    elapsed = time.time() - start

    print(f"Citation matching elapsed for N=100, M=10000: {elapsed:.4f}s")
    assert elapsed < 1.5, f"Citation matching too slow: {elapsed:.4f}s"


if __name__ == "__main__":
    test_citation_matching_performance()
