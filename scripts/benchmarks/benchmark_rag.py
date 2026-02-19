"""
Ragas를 활용한 RAG 시스템 성능 벤치마크 스크립트.
로컬 Ollama 모델을 평가관(Judge)으로 사용합니다.
"""

import asyncio
import logging
import os
import sys
import pandas as pd
from datetime import datetime
from typing import List

# src 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), \"..\", \"..\", \"src\"))

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_benchmark(file_path: str, file_name: str, questions: List[str], ground_truths: List[str]):
    """
    RAG 시스템에 대한 벤치마크 테스트를 수행합니다.
    """
    logger.info(f"벤치마크 시작: {file_name}")
    
    # 1. RAG 시스템 준비
    rag = RAGSystem(session_id="benchmark_session")
    embedder = ModelManager.get_embedder()
    llm = ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    
    # 문서 로드 및 인덱싱
    await rag.load_document(file_path, file_name, embedder)
    
    # 2. 결과 수집
    results = []
    for q, gt in zip(questions, ground_truths):
        logger.info(f"질문 처리 중: {q}")
        response = await rag.aquery(q, llm=llm)
        
        results.append({
            "user_input": q,
            "response": response["response"],
            "retrieved_contexts": [doc.page_content for doc in response.get("documents", [])],
            "reference": gt
        })
    
    # 3. Ragas 평가 데이터셋 생성
    # EvaluationService는 {"query": ..., "response": ..., "context": ...} 형식을 기대함
    eval_data = []
    for r in results:
        eval_data.append({
            "query": r["user_input"],
            "response": r["response"],
            "context": r["retrieved_contexts"]
        })
    
    # 4. EvaluationService를 통한 평가 실행
    logger.info("EvaluationService를 통한 Ragas 평가 시작...")
    eval_service = EvaluationService()
    summary, report_path = await eval_service.run_evaluation(eval_data, report_prefix="benchmark_report")
    
    logger.info(f"벤치마크 완료! 결과 저장됨: {report_path}")
    print("\n=== 벤치마크 결과 요약 ===")
    print(summary)
    
    return summary

if __name__ == "__main__":
    # 생성된 테스트셋 로드
    testset_path = "tests/data/testset_2201.csv"
    pdf_path = "tests/data/2201.07520v1.pdf"
    
    if not os.path.exists(testset_path):
        logger.error(f"테스트셋 파일이 없습니다: {testset_path}")
    else:
        df = pd.read_csv(testset_path)
        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()
        
        asyncio.run(run_benchmark(pdf_path, "2201.07520v1.pdf", questions, ground_truths))

