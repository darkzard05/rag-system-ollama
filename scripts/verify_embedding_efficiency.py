
import asyncio
import time
import os
import sys
from unittest.mock import MagicMock

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.rag_core import build_rag_pipeline
from core.model_loader import ModelManager
from core.session import SessionManager

async def benchmark_indexing():
    print("=== RAG 인덱싱 효율성 테스트 (v2) ===")
    
    # 1. 준비: 임베딩 모델 로드
    embedder = ModelManager.get_embedder()
    
    # 호출 횟수를 기록하기 위한 카운터
    call_stats = {"count": 0}
    
    # 기존 메서드 보관
    real_embed_documents = embedder.embed_documents
    
    # 래퍼 함수 정의
    def wrapped_embed_documents(texts):
        call_stats["count"] += 1
        print(f"  > [DEBUG] embed_documents 호출됨! (청크 수: {len(texts)})")
        return real_embed_documents(texts)
    
    # 메서드 교체 (Pydantic 모델 보호 우회)
    object.__setattr__(embedder, "embed_documents", wrapped_embed_documents)
    
    # 2. PDF 로딩 단계를 건너뛰고 가짜 문서 반환하도록 모킹
    import core.rag_core
    from langchain_core.documents import Document
    
    # 충분한 양의 텍스트를 가진 샘플 문서
    sample_docs = [
        Document(page_content="테스트 문장입니다. " * 50, metadata={"source": "test.pdf", "page": 1})
    ]
    core.rag_core._load_pdf_docs = MagicMock(return_value=sample_docs)

    # 3. 인덱싱 실행
    start_time = time.time()
    session_id = f"test_{int(start_time)}"
    SessionManager.init_session(session_id=session_id)
    
    # 캐시 무효화를 위해 설정 해시를 바꿀 수 없으므로, 파일명을 고유하게 함
    unique_filename = f"test_{int(start_time)}.pdf"
    
    print(f"인덱싱 시작... (파일명: {unique_filename})")
    msg, cache_used = await build_rag_pipeline(
        uploaded_file_name=unique_filename,
        file_path="fake.pdf",
        embedder=embedder,
        session_id=session_id
    )
    
    end_time = time.time()
    
    # 4. 결과 출력
    print("\n--- 테스트 결과 ---")
    print(f"임베딩 함수 호출 총 횟수: {call_stats['count']}회")
    print(f"인덱싱 완료 메시지: {msg}")
    print(f"소요 시간: {end_time - start_time:.2f}초")
    
    if call_stats['count'] == 1:
        print("결과: ✅ [성공] 임베딩이 중복 없이 단 1회만 호출되었습니다!")
    else:
        print(f"결과: ❌ [실패] 임베딩이 {call_stats['count']}회 호출되었습니다.")

if __name__ == "__main__":
    try:
        asyncio.run(benchmark_indexing())
    except Exception as e:
        import traceback
        traceback.print_exc()
