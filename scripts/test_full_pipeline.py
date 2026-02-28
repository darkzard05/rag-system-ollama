import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from core.session import SessionManager
from common.config import DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL
from common.logging_config import setup_logging

async def run_full_pipeline_test():
    # [동기화] 실제 앱과 동일한 로깅 설정 적용
    setup_logging(log_level="INFO")
    
    print("\n" + "=" * 50)
    print("[E2E] RAG Pipeline Integration Test (App Synchronized)")
    print("설명: 실제 앱(UI/API)과 동일한 호출 방식으로 파이프라인을 검증합니다.")
    print("=" * 50)

    # 1. 세션 초기화 (실제 앱과 동일한 라이프사이클)
    session_id = f"test-session-{int(datetime.now().timestamp())}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    # 2. 모델 준비
    print("\n1. Preparing Embedding Model...")
    try:
        # 임베더는 인덱싱을 위해 명시적으로 필요
        embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 3. 문서 인덱싱
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = os.path.basename(test_pdf)
    print(f"\n2. Indexing Document: {file_name}")
    
    start_time = asyncio.get_event_loop().time()
    # build_pipeline 내부에서 ResourcePool 등록 및 세션 정보 저장이 일어남
    msg, cache_used = await rag.build_pipeline(test_pdf, file_name, embedder)
    load_time = asyncio.get_event_loop().time() - start_time
    print(f"   Result: {msg} (Cache: {cache_used}) | Time: {load_time:.2f}s")

    # 4. 통합 인터페이스 질의 (실제 앱 호출 방식과 100% 일치)
    test_cases = [
        "CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤 차이가 있어?"
    ]

    print("\n3. Running App-Synchronized Queries...")
    for i, test_query in enumerate(test_cases):
        print(f"   [{i+1}/{len(test_cases)}] Querying: '{test_query[:50]}...'")
        
        start_t = asyncio.get_event_loop().time()
        # [리팩토링 반영] llm 객체 대신 모델 이름만 전달하여 내부 자동 관리 유도
        result = await rag.aquery(test_query, model_name=DEFAULT_OLLAMA_MODEL)
        q_time = asyncio.get_event_loop().time() - start_t
        
        response = result.get("response", "")
        context = result.get("context", "")
        
        print(f"   -> Done ({q_time:.2f}s)")
        print(f"   -> Response Length: {len(response)} chars")
        print(f"   -> Context Chunks Used: {len(result.get('documents', []))}")

        # 기능적 검증
        if len(response) > 100 and len(result.get('documents', [])) > 0:
            print(f"   ✅ [PASS] Query {i+1} functional check successful.")
        else:
            print(f"   ❌ [FAIL] Query {i+1} produced suspicious output.")

    print(f"\n4. Pipeline Test Finished (Total Queries: {len(test_cases)})")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_full_pipeline_test())
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
