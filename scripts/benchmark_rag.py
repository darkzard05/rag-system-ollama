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
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

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
    dataset = EvaluationDataset.from_list(results)
    
    # 4. 평가 모델(Judge) 설정 (Ollama 기반)
    # 평가에는 조금 더 파라미터가 큰 모델(예: llama3.1:8b)을 권장하지만, 일단 기본 모델 사용
    judge_llm = LangchainLLMWrapper(ChatOllama(model=DEFAULT_OLLAMA_MODEL, base_url=OLLAMA_BASE_URL))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL))
    
    # 5. 평가 실행
    logger.info("Ragas 평가 지표 계산 중...")
    eval_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    
    # 6. 결과 저장 및 출력
    df = eval_result.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"reports/benchmark_result_{timestamp}.csv"
    
    os.makedirs("reports", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    logger.info(f"벤치마크 완료! 결과 저장됨: {output_path}")
    print("\n=== 벤치마크 결과 요약 ===")
    print(eval_result)
    
    return eval_result

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
