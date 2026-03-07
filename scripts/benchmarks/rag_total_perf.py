"""
Ragas를 활용한 RAG 시스템 성능 벤치마크 스크립트.
로컬 Ollama 모델을 평가관(Judge)으로 사용합니다.
"""

import asyncio
import logging
import os
import sys
import time
import pandas as pd
from datetime import datetime
from typing import List

# src 디렉토리를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL
from services.evaluation_service import EvaluationService
from services.monitoring.performance_monitor import get_performance_monitor, OperationType

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

async def run_benchmark(file_path: str, file_name: str, questions: List[str], ground_truths: List[str]):
    """
    RAG 시스템에 대한 벤치마크 테스트를 수행합니다.
    """
    logger.info(f"벤치마크 시작: {file_name}")
    
    # 1. RAG 시스템 준비
    rag = RAGSystem(session_id="benchmark_session")
    
    start_time = time.perf_counter()
    logger.info("문서 로드 및 인덱싱 시작...")
    # [수정] await 명시
    embedder = await ModelManager.get_embedder()
    await rag.load_document(file_path, file_name, embedder)
    index_time = time.perf_counter() - start_time
    logger.info(f"인덱싱 완료: {index_time:.2f}s")
    
    # 2. 결과 수집 (최대 3개로 제한)
    results = []
    qa_times = []
    
    for i, (q, gt) in enumerate(zip(questions[:3], ground_truths[:3])):
        logger.info(f"[{i+1}/3] 질문 처리 중: {q}")
        
        qa_start = time.perf_counter()
        # RAG 쿼리 실행
        response = await rag.aquery(q, model_name=DEFAULT_OLLAMA_MODEL)
        qa_duration = time.perf_counter() - qa_start
        qa_times.append(qa_duration)
        
        results.append({
            "user_input": q,
            "response": response["response"],
            "retrieved_contexts": [doc.page_content for doc in response.get("relevant_docs", response.get("documents", []))],
            "reference": gt
        })
        logger.info(f"질문 처리 완료: {qa_duration:.2f}s")
    
    # 성능 통계 출력
    avg_qa_time = sum(qa_times) / len(qa_times) if qa_times else 0
    print("\n" + "="*50)
    print("       RAG PIPELINE PERFORMANCE REPORT")
    print("="*50)
    print(f"Indexing Time:  {index_time:.2f}s")
    print(f"Avg QA Latency: {avg_qa_time:.2f}s")
    
    # Monitor 통계 출력 (OperationType 활용)
    report = monitor.generate_report()
    ops = report.get("operations", {})
    for op_name, stats in ops.items():
        dur = stats.get("duration", {})
        print(f"[{op_name:20}] Avg: {dur.get('avg', 0):.4f}s | Max: {dur.get('max', 0):.4f}s")
    print("="*50)

    # 3. Ragas 평가 생략 (병목 파악 목적이므로 시간 단축)
    return results

if __name__ == "__main__":
    testset_path = "tests/data/testset_2201.csv"
    pdf_path = "tests/data/2201.07520v1.pdf"
    
    if not os.path.exists(testset_path):
        logger.error(f"테스트셋 파일이 없습니다: {testset_path}")
    else:
        df = pd.read_csv(testset_path)
        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()
        
        asyncio.run(run_benchmark(pdf_path, "2201.07520v1.pdf", questions, ground_truths))

