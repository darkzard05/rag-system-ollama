import asyncio
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SystemTest")


async def test_full_pipeline():
    logger.info("=== RAG 파이프라인 전체 테스트 시작 ===")

    # 1. 모델 로드
    logger.info(f"1. 임베딩 및 LLM({DEFAULT_OLLAMA_MODEL}) 로드 중...")
    embedder = load_embedding_model()
    llm = load_llm(model_name=DEFAULT_OLLAMA_MODEL)

    # 2. RAG 시스템 초기화 및 문서 로드
    test_pdf = "tests/2201.07520v1.pdf"
    if not os.path.exists(test_pdf):
        logger.error(f"테스트 파일이 없습니다: {test_pdf}")
        return

    rag = RAGSystem(session_id="test_session")
    logger.info(f"2. 문서 인덱싱 시작: {test_pdf}")
    # load_document는 RAGSystem 클래스에 구현될 예정입니다.
    msg, cache_used = await rag.load_document(test_pdf, "test_paper.pdf", embedder)
    logger.info(f"결과: {msg} (캐시 사용: {cache_used})")

    # 3. 질문 테스트
    query = "What is the main topic of this paper?"
    logger.info(f"3. 질문 생성 및 답변 대기: '{query}'")

    start_time = asyncio.get_event_loop().time()
    try:
        # aquery는 RAGSystem 클래스에 구현될 예정입니다.
        response = await rag.aquery(query, llm=llm)
        elapsed = asyncio.get_event_loop().time() - start_time

        logger.info(f"=== 답변 생성 완료 ({elapsed:.2f}s) ===")
        logger.info(f"답변 요약: {response['response'][:200]}...")

        # 4. 컨텍스트 검증 (문서가 검색되었는지 확인)
        docs = response.get("documents", [])
        logger.info(f"검증: 검색된 문서 수 = {len(docs)}")

        if len(docs) > 0:
            logger.info("✅ 관련 문서가 성공적으로 검색되었습니다.")
        else:
            logger.warning("❌ 검색된 문서가 없습니다.")

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
