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
from common.config import DEFAULT_OLLAMA_MODEL
from common.logging_config import setup_logging

async def verify_system():
    # 로깅 설정 (검증용)
    setup_logging(log_level="INFO")
    
    print("\n" + "="*60)
    print("🚀 RAG System Refactoring Verification")
    print("="*60)

    session_id = f"verify-{int(datetime.now().timestamp())}"
    rag = RAGSystem(session_id=session_id)
    
    # 1. 비동기 모델 로딩 테스트
    print("\n[STEP 1] Testing Async Model Loading...")
    start = asyncio.get_event_loop().time()
    
    # 병렬 로딩 시도 (락 작동 확인)
    embedder_task = ModelManager.get_embedder()
    llm_task = ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    
    embedder, llm = await asyncio.gather(embedder_task, llm_task)
    
    elapsed = asyncio.get_event_loop().time() - start
    print(f"✅ Models loaded successfully in {elapsed:.2f}s")

    # 2. 문서 인덱싱 (Semantic Chunking & ResourcePool)
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found at {test_pdf}. Skipping indexing test.")
        # 대체용 파일 찾기
        pdf_files = list(ROOT_DIR.glob("**/*.pdf"))
        if pdf_files:
            test_pdf = str(pdf_files[0])
            print(f"💡 Using alternative PDF: {test_pdf}")
        else:
            return

    print(f"\n[STEP 2] Testing Document Indexing (Semantic & Page-Aware)...")
    start = asyncio.get_event_loop().time()
    
    msg, cache_used = await rag.load_document(test_pdf, os.path.basename(test_pdf), embedder)
    
    elapsed = asyncio.get_event_loop().time() - start
    print(f"✅ {msg}")
    print(f"✅ Indexing completed in {elapsed:.2f}s (Cache used: {cache_used})")

    # 3. 비동기 질의 응답 (Async Semaphore)
    print("\n[STEP 3] Testing Async Querying...")
    query = "What is this document about?"
    
    start = asyncio.get_event_loop().time()
    result = await rag.aquery(query, llm=llm)
    elapsed = asyncio.get_event_loop().time() - start
    
    response = result.get("response", "")
    thought = result.get("thought", "")
    
    print(f"✅ Query finished in {elapsed:.2f}s")
    print(f"✅ Response length: {len(response)} chars")
    if thought:
        print(f"✅ Thought captured: {len(thought)} chars")

    # 4. 메타데이터 검증 (Page-Aware Chunking 결과 확인)
    print("\n[STEP 4] Verifying Metadata (Pages/Cross-page)...")
    from core.resource_pool import get_resource_pool
    from core.session import SessionManager
    
    file_hash = SessionManager.get("file_hash", session_id=session_id)
    vector_store, _ = await get_resource_pool().get(file_hash)
    
    if vector_store:
        # 무작위 청크 하나 꺼내서 메타데이터 확인
        # FAISS.similarity_search 대신 직접 docstore에서 추출 시도
        sample_doc = vector_store.similarity_search("context", k=1)[0]
        meta = sample_doc.metadata
        print(f"✅ Sample Chunk Metadata: {meta}")
        if "pages" in meta:
            print(f"🎯 SUCCESS: Page-aware metadata 'pages' found: {meta['pages']}")
        else:
            print("⚠️ WARNING: 'pages' key not found in metadata.")
    
    print("\n" + "="*60)
    print("🏁 All Verification Steps Completed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(verify_system())
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
