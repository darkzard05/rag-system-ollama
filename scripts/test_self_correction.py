
import asyncio
import logging
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from core.session import SessionManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SelfCorrectionTest")

async def run_test():
    session_id = "test_self_correction"
    rag = RAGSystem(session_id=session_id)
    
    # 1. 문서 인덱싱
    pdf_path = "tests/data/2201.07520v1.pdf"
    logger.info(f"인덱싱 시작: {pdf_path}")
    
    # 임베딩 모델 로드 (HuggingFaceEmbeddings 사용)
    from langchain_huggingface import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    
    msg, is_cached = await rag.build_pipeline(pdf_path, "2201.07520v1.pdf", embedder)
    logger.info(f"결과: {msg} (캐시 사용: {is_cached})")
    
    # 2. 질문 수행 (일부러 모호하거나 구체적인 정보 요청)
    # 문서가 아마도 논문일 것으로 보이므로, 초록이나 서론에 없는 구체적인 구현 디테일을 물어봅니다.
    query = "이 논문에서 제안한 알고리즘의 구체적인 Python 구현 코드 예시가 포함되어 있나요?"
    
    logger.info(f"질문 수행: {query}")
    
    # 3. 답변 생성 (스트리밍 대신 Full Response 사용)
    result = await rag.aquery(query)
    
    print("\n" + "="*50)
    print(f"질문: {query}")
    print("-" * 50)
    print(f"사고 과정:\n{result.get('thought', '사고 과정 없음')}")
    print("-" * 50)
    print(f"답변:\n{result.get('response', '답변 없음')}")
    print("="*50)
    
    # 4. 상태 로그 확인 (워크플로우 추적)
    print("\n[상태 로그 - 워크플로우 추적]")
    logs = rag.get_status()
    for log in logs:
        print(f" -> {log}")

if __name__ == "__main__":
    asyncio.run(run_test())
