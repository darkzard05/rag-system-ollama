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
from common.config import DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL
from common.logging_config import setup_logging

async def verify_system():
    # 로깅 설정 (검증용)
    setup_logging(log_level="INFO")
    
    print("\n" + "="*60)
    print("🚀 RAG System Refactoring Verification (Integrated Interface)")
    print("="*60)

    session_id = f"verify-{int(datetime.now().timestamp())}"
    # 세션 ID를 지정하여 RAGSystem 인스턴스 생성
    rag = RAGSystem(session_id=session_id)
    
    # 1. 모델 로딩 (임베더는 문서 로드를 위해 필요)
    print("\n[STEP 1] Preparing Embedding Model...")
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    print(f"✅ Embedding model '{DEFAULT_EMBEDDING_MODEL}' ready.")

    # 2. 문서 인덱싱
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found at {test_pdf}.")
        return

    print(f"\n[STEP 2] Testing Document Indexing...")
    # build_pipeline 내부에서 ResourcePool에 리소스를 등록함
    msg, cache_used = await rag.build_pipeline(test_pdf, os.path.basename(test_pdf), embedder)
    
    print(f"✅ {msg} (Cache used: {cache_used})")

    # 3. 통합 인터페이스 질의 테스트 (aquery)
    # 이제 aquery 내부에서 ModelManager를 통해 LLM을 가져오고 ResourcePool에서 리트리버를 가져옴
    print("\n[STEP 3] Testing Integrated Query Interface (aquery)...")
    query = "What is the main topic of this paper?"
    
    start = asyncio.get_event_loop().time()
    # [핵심] llm 객체 대신 model_name만 전달
    result = await rag.aquery(query, model_name=DEFAULT_OLLAMA_MODEL)
    elapsed = asyncio.get_event_loop().time() - start
    
    response = result.get("response", "")
    thought = result.get("thought", "")
    
    print(f"✅ Query finished in {elapsed:.2f}s")
    print(f"✅ Response (first 100 chars): {response[:100]}...")
    if thought:
        print(f"✅ Thought captured (first 100 chars): {thought[:100]}...")

    # 4. 리소스 풀 및 세션 연동 확인
    print("\n[STEP 4] Verifying ResourcePool & Session Linkage...")
    from core.resource_pool import get_resource_pool
    from core.session import SessionManager
    
    file_hash = SessionManager.get("file_hash", session_id=session_id)
    print(f"📊 Session File Hash: {file_hash[:8]}...")
    
    vector_store, bm25 = await get_resource_pool().get(file_hash)
    if vector_store and bm25:
        print("✅ SUCCESS: ResourcePool correctly holds both VectorStore and BM25 for this session.")
    else:
        print("❌ FAILURE: ResourcePool is missing resources for this session.")
        if not vector_store: print("   - VectorStore is None")
        if not bm25: print("   - BM25 is None")

    print("\n" + "="*60)
    print("🏁 Refactoring Verification Completed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(verify_system())
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
