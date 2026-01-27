
import asyncio
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import OLLAMA_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SystemTest")

async def test_full_pipeline():
    logger.info("=== RAG 파이프라인 전체 테스트 시작 ===")
    
    # 1. 모델 로드
    logger.info(f"1. 임베딩 및 LLM({OLLAMA_MODEL_NAME}) 로드 중...")
    embedder = load_embedding_model()
    llm = load_llm(model_name=OLLAMA_MODEL_NAME)
    
    # 2. RAG 시스템 초기화 및 문서 로드
    # 테스트용 PDF 경로 (가장 최근 로그에 찍힌 파일 중 하나 사용 또는 샘플)
    test_pdf = "tests/2201.07520v1.pdf" # 기존 테스트 폴더의 PDF 사용
    if not os.path.exists(test_pdf):
        logger.error(f"테스트 파일이 없습니다: {test_pdf}")
        return

    rag = RAGSystem(session_id="test_session")
    logger.info(f"2. 문서 인덱싱 시작: {test_pdf}")
    msg, cache_used = rag.load_document(test_pdf, "test_paper.pdf", embedder)
    logger.info(f"결과: {msg} (캐시 사용: {cache_used})")
    
    # 3. 질문 테스트
    query = "What is the main topic of this paper?"
    logger.info(f"3. 질문 생성 및 답변 대기: '{query}'")
    
    start_time = asyncio.get_event_loop().time()
    try:
        # 스트리밍 이벤트를 관측하기 위해 aquery 호출
        response = await rag.aquery(query, llm=llm)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"=== 답변 생성 완료 ({elapsed:.2f}s) ===")
        logger.info(f"답변 요약: {response['response'][:200]}...")
        
        # 4. 컨텍스트 크기 검증
        context_len = len(response.get("context", ""))
        logger.info(f"검증: 전달된 컨텍스트 길이 = {context_len}자 (제한치 2500자 이내 여부 확인)")
        
        if context_len <= 2600: # 약간의 여유분 포함
            logger.info("✅ 컨텍스트 제한이 성공적으로 적용되었습니다.")
        else:
            logger.warning("❌ 컨텍스트 제한이 제대로 작동하지 않았습니다.")

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
