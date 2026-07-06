import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가 (src 폴더를 모듈 경로로 인식하게 함)
ROOT_DIR = Path(__file__).parent.parent.absolute()
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from src.core.rag_core import RAGSystem
from src.services.evaluation_service import EvaluationService
from src.core.model_loader import ModelManager
from src.common.config import DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL
from src.core.session import SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1. Load Golden Set
    golden_set_path = ROOT_DIR / "tests/data/golden_set.json"
    if not golden_set_path.exists():
        logger.error(f"Golden set file not found at: {golden_set_path}")
        return

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    # 2. Initialize RAG System and Eval Service
    # 세션 초기화
    session_id = f"eval-session-{int(datetime.now().timestamp())}"
    SessionManager.init_session(session_id=session_id)
    
    rag_system = RAGSystem(session_id=session_id)
    eval_service = EvaluationService()
    
    # 테스트용 PDF 경로
    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf") 
    
    # 임베더 로드 및 파이프라인 구축
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    await rag_system.build_pipeline(pdf_path, "eval_doc", embedder)

    data_points = []
    logger.info(f"Evaluating {len(golden_set)} queries...")

    for item in golden_set:
        query = item["query"]
        logger.info(f"Querying: {query}")
        
        # 응답 및 컨텍스트 수집
        response_text = ""
        retrieved_docs = []
        
        try:
            # astream_events를 통해 중간 단계(retrieve)의 결과물 수집
            async for event in rag_system.astream_events(query, model_name=DEFAULT_OLLAMA_MODEL):
                if event["event"] == "on_chain_end" and event["name"] == "retrieve":
                    # retrieve 노드의 출력을 저장
                    retrieved_docs = event["data"]["output"].get("relevant_docs", [])
                elif event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    # chunk content 추출 (객체인 경우 .content, 문자열인 경우 그대로 사용)
                    content = getattr(chunk, 'content', chunk) if chunk else ""
                    if content:
                        response_text += content
        except Exception as e:
            logger.error(f"Error during query {query}: {e}")
            continue

        # RAGAS 평가를 위해 Document 객체 리스트를 텍스트 리스트로 변환
        processed_contexts = [d.page_content for d in retrieved_docs]

        data_points.append({
            "query": query,
            "response": response_text,
            "context": processed_contexts
        })

    # 3. RAGAS 평가 실행
    if not data_points:
        logger.error("No successful responses collected. Evaluation aborted.")
        return

    summary, report_path = await eval_service.run_evaluation(
        data_points, 
        report_prefix="performance_optimized_eval"
    )

    logger.info(f"Evaluation Complete. Summary: {summary}")
    logger.info(f"Report saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
