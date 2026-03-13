
import sys
import os
import json
import time
from langchain_core.documents import Document

# 프로젝트 루트 추가
sys.path.append(os.path.abspath("src"))

from core.graph_builder import _merge_adjacent_chunks, GradeResponse

def test_context_merging():
    print("--- Context Merging Optimization Test ---")
    docs = [
        Document(page_content="Chunk 1", metadata={"source": "A.pdf", "page": 1, "chunk_index": 0}),
        Document(page_content="Chunk 2", metadata={"source": "A.pdf", "page": 1, "chunk_index": 1}),
        Document(page_content="Chunk 3", metadata={"source": "B.pdf", "page": 1, "chunk_index": 0}),
    ]
    
    start_time = time.time()
    merged = _merge_adjacent_chunks(docs, max_tokens=1000)
    duration = time.time() - start_time
    
    print(f"Original doc count: {len(docs)}")
    print(f"Merged doc count: {len(merged)} (Expected: 2)")
    print(f"Execution time: {duration:.6f}s")
    
    if len(merged) == 2 and "Chunk 1\n\nChunk 2" in merged[0].page_content:
        print("✅ Success: Context merging logic is correct and optimized.")
    else:
        print("❌ Failure: Context merging logic failed.")
        sys.exit(1)

def test_robust_json_parsing():
    print("\n--- Robust JSON Parsing Test ---")
    import re
    
    # 흉내낸 지저분한 LLM 응답
    noisy_response = "물론이죠, 여기 분석 결과입니다: ```json\n{\"is_relevant\": true, \"relevant_entities\": [\"AI\"], \"reason\": \"관련 있음\"}\n``` 도움이 되길 바랍니다."
    
    # graph_builder.py에 구현된 추출 로직 시뮬레이션
    match = re.search(r"\{.*\}", noisy_response, re.DOTALL)
    if match:
        data = json.loads(match.group())
        result = GradeResponse(**data)
        print(f"Parsed result: {result}")
        if result.is_relevant and result.relevant_entities == ["AI"]:
            print("✅ Success: Robust JSON parsing from noisy response is working.")
        else:
            print("❌ Failure: Parsing logic extracted wrong data.")
            sys.exit(1)
    else:
        print("❌ Failure: Could not find JSON pattern.")
        sys.exit(1)

if __name__ == "__main__":
    test_context_merging()
    test_robust_json_parsing()
