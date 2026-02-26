import os
import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL
from common.logging_config import setup_logging

async def run_full_pipeline_test():
    # 로깅 설정 초기화
    setup_logging(log_level="INFO", log_file=ROOT_DIR / "logs" / "test_e2e.log")
    
    print("""
[E2E] RAG Pipeline Integration Test Started
설명: 전체 파이프라인의 기능적 연결성을 검증합니다. (평가 제외)
""")
    
    session_id = f"test-session-{int(datetime.now().timestamp())}"
    rag = RAGSystem(session_id=session_id)
    
    print("1. Loading Models...")
    try:
        embedder = load_embedding_model()
        llm = load_llm(DEFAULT_OLLAMA_MODEL)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    print(f"2. Indexing Document: {os.path.basename(test_pdf)}")
    
    start_time = asyncio.get_event_loop().time()
    msg, cache_used = await rag.load_document(test_pdf, "2201.07520v1.pdf", embedder)
    load_time = asyncio.get_event_loop().time() - start_time
    print(f"   Result: {msg} (Cache: {cache_used}) | Time: {load_time:.2f}s")

    # 기능 검증용 테스트 쿼리
    test_cases = [
        "CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤 차이가 있어?"
    ]
    
    print("\n3. Running Test Queries...")
    for i, test_query in enumerate(test_cases):
        print(f"   [{i+1}/{len(test_cases)}] Querying: '{test_query[:50]}...'")
        
        start_t = asyncio.get_event_loop().time()
        result = await rag.aquery(test_query, llm=llm)
        q_time = asyncio.get_event_loop().time() - start_t
        
        # 가벼운 검증 (Sanity Check)
        response = result.get("response", "")
        context = result.get("context", "")
        
        print(f"   -> Done ({q_time:.2f}s)")
        print(f"   -> Response Length: {len(response)} chars")
        print(f"   -> Context Chunks Used: {context.count('### [자료')}")

        if len(response) > 50 and context:
            print(f"   ✅ [PASS] Query {i+1} functional check successful.")
        else:
            print(f"   ❌ [FAIL] Query {i+1} produced suspicious output.")

    print(f"\n4. Pipeline Test Finished (Total Queries: {len(test_cases)})")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(run_full_pipeline_test())