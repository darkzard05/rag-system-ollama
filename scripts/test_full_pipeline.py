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
from services.evaluation_service import EvaluationService

async def run_evaluation(data_points: list[dict]):
    """EvaluationService를 사용하여 생성된 답변의 품질을 평가합니다."""
    print("\n" + "=" * 50)
    print("[Ragas] Starting Automatic Quality Evaluation...")
    
    try:
        eval_service = EvaluationService()
        summary, report_path = await eval_service.run_evaluation(data_points, report_prefix="e2e_eval_report")

        print(f"[Ragas] Evaluation complete. Scores: {summary}")
        print(f"[Ragas] Detailed report saved to: {report_path}")
        print("=" * 50)

    except Exception as e:
        print(f"[Ragas] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

async def run_full_pipeline_test():
    # 로깅 설정 초기화
    setup_logging(log_level="INFO", log_file=ROOT_DIR / "logs" / "test_e2e.log")
    
    print("[E2E] RAG Pipeline Integration Test Started")
    
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

    # 평가를 위한 테스트 쿼리 세트 (빠른 검증을 위해 1개로 제한)
    test_cases = [
        "CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤 차이가 있어?"
    ]
    
    captured_data = []

    print("\n3. Running Test Queries & Collecting Data...")
    for i, test_query in enumerate(test_cases):
        print(f"   [{i+1}/{len(test_cases)}] Querying: '{test_query[:50]}...'")
        
        start_t = asyncio.get_event_loop().time()
        result = await rag.aquery(test_query, llm=llm)
        q_time = asyncio.get_event_loop().time() - start_t
        
        print(f"   -> Done ({q_time:.2f}s)")
        
        captured_data.append({
            "query": test_query,
            "response": result.get("response", ""),
            "context": result.get("context", "")
        })

    print(f"\n4. Pipeline Test Finished (Total Queries: {len(test_cases)})")
    
    # 5. [추가] 즉시 평가 실행
    await run_evaluation(captured_data)

if __name__ == "__main__":
    asyncio.run(run_full_pipeline_test())