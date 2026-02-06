import os
import sys
import time
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Any

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL, AVAILABLE_EMBEDDING_MODELS
from core.session import SessionManager
from core.query_optimizer import RAGQueryOptimizer

# 테스트 환경 설정
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
MODEL_NAME = AVAILABLE_EMBEDDING_MODELS[0]
TEST_QUERY = "What are the main contributions of the CM3 model?"

async def run_benchmark(device_type: str):
    print(f"\n>>> Running Benchmark on: [{device_type.upper()}]")
    
    # 1. 초기화 및 모델 로드
    os.environ["EMBEDDING_DEVICE"] = device_type
    # 모델 인스턴스 초기화 (캐시 무효화)
    ModelManager._instances.clear()
    
    session_id = f"bench-{device_type}-{int(time.time())}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)
    
    # 모델 로드 시간 측정
    start = time.time()
    embedder = ModelManager.get_embedder(MODEL_NAME)
    llm = ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    load_time = time.time() - start
    print(f"   [Step 0] Model Loading: {load_time:.2f}s")

    # 2. Phase 1: Document Indexing (Preprocessing + Embedding + FAISS)
    start = time.time()
    msg, cache_used = await rag.load_document(TEST_PDF, "bench.pdf", embedder)
    indexing_time = time.time() - start
    print(f"   [Step 1] Full Indexing: {indexing_time:.2f}s (Cache Used: {cache_used})")

    # 3. Phase 2: Intent Classification
    start = time.time()
    for _ in range(5):
        _ = await RAGQueryOptimizer.classify_intent(TEST_QUERY, llm)
    intent_latency = (time.time() - start) / 5 * 1000
    print(f"   [Step 2] Intent Classification: {intent_latency:.2f}ms")

    # 4. Phase 3: Vector Retrieval
    faiss_ret = SessionManager.get("faiss_retriever", session_id=session_id)
    start = time.time()
    for _ in range(10):
        _ = await faiss_ret.ainvoke(TEST_QUERY)
    retrieval_latency = (time.time() - start) / 10 * 1000
    print(f"   [Step 3] Vector Retrieval (K=20): {retrieval_latency:.2f}ms")

    return {
        "device": device_type,
        "indexing": indexing_time,
        "intent": intent_latency,
        "retrieval": retrieval_latency
    }

async def main():
    print("============================================================")
    print("🔍 Comprehensive Embedding Performance Benchmark")
    print("============================================================")
    
    results = []
    
    if torch.cuda.is_available():
        res_gpu = await run_benchmark("cuda")
        results.append(res_gpu)
    else:
        print("\n[SKIP] CUDA not available.")

    res_cpu = await run_benchmark("cpu")
    results.append(res_cpu)

    if len(results) >= 1:
        print("\n============================================================")
        print("📊 Performance Results Summary")
        print("------------------------------------------------------------")
        for res in results:
            d = res["device"].upper()
            print(f"[{d}] Indexing: {res['indexing']:.2f}s | Intent: {res['intent']:.2f}ms | Retrieval: {res['retrieval']:.2f}ms")
        
        if len(results) == 2:
            gpu, cpu = results[0], results[1]
            print("------------------------------------------------------------")
            print(f"Indexing Speedup: {cpu['indexing']/gpu['indexing']:.2f}x")
            print(f"Intent Speedup: {cpu['intent']/gpu['intent']:.2f}x")
            print(f"Retrieval Speedup: {cpu['retrieval']:.2f}x")
        print("============================================================")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())