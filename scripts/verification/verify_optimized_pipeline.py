import os
import sys
import asyncio
import time
from pathlib import Path
import numpy as np

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL
from core.session import SessionManager

async def run_optimized_pipeline_test():
    print("\n============================================================")
    print("🚀 Optimized RAG Pipeline Full-Stack Verification")
    print("============================================================")
    
    session_id = f"verify-opt-{int(time.time())}"
    rag = RAGSystem(session_id=session_id)
    
    # 1. 모델 로드 (가속 엔진 활성화 확인)
    print("\n[STEP 1] Model Loading")
    start = time.time()
    embedder = ModelManager.get_embedder()
    llm = ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    load_time = time.time() - start
    
    device = SessionManager.get("current_embedding_device", session_id=session_id)
    print(f"✅ Models Loaded in {load_time:.2f}s")
    print(f"🔹 Embedding Device: {device}")

    # 2. 문서 인덱싱 (최적화된 청킹 및 프루닝 확인)
    print("\n[STEP 2] Document Indexing")
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    
    start = time.time()
    msg, cache_used = await rag.load_document(test_pdf, "CM3_Paper.pdf", embedder)
    indexing_time = time.time() - start
    
    print(f"✅ Indexing Complete in {indexing_time:.2f}s (Cache Used: {cache_used})")
    
    # 3. 최적화된 의도 분석 테스트 (Vectorized Logic)
    print("\n[STEP 3] Intent Classification (Vectorized)")
    from core.query_optimizer import RAGQueryOptimizer
    
    test_queries = [
        "Hi, how are you?",  # GREETING
        "Who are the authors of this paper?",  # FACTOID
        "Summarize the main contribution of CM3 in detail."  # RESEARCH
    ]
    
    for q in test_queries:
        start_q = time.time()
        intent = await RAGQueryOptimizer.classify_intent(q, llm)
        q_lat = (time.time() - start_q) * 1000
        print(f"🔹 Query: '{q[:30]}...' -> Intent: {intent} ({q_lat:.1f}ms)")

    # 4. 전체 질의 응답 (RRF 및 최적화된 리트리버)
    print("\n[STEP 4] Full RAG Query Execution")
    final_query = "Explain the difference between CM3 and previous causal-only models."
    
    start = time.time()
    result = await rag.aquery(final_query, llm=llm)
    total_lat = time.time() - start
    
    print(f"✅ Query Finished in {total_lat:.2f}s")
    print("\n------------------------------ [RESULT] ------------------------------")
    print(f"Route: {result.get('route_decision')}")
    print(f"Context: {len(result.get('documents', []))} document blocks retrieved.")
    print("\n[Final Answer Preview]")
    ans = result.get('response', '')
    print(ans[:500] + ("..." if len(ans) > 500 else ""))
    print("----------------------------------------------------------------------")

if __name__ == "__main__":
    # Windows loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_optimized_pipeline_test())
