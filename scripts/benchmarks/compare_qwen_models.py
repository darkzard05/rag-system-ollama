"""
Qwen 3.5 vs Qwen 3 (4B 모델) 성능 및 품질 1:1 비교 벤치마크.
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def run_model_comparison(file_path: str, file_name: str, questions: List[str]):
    """
    두 모델을 번갈아가며 호출하여 성능 지표를 수집합니다.
    """
    models = ["qwen3:4b-instruct-2507-q4_K_M", "qwen3.5:4b"]
    results = []
    
    # 1. RAG 시스템 준비 (임베딩 및 인덱싱은 1회만 수행)
    rag = RAGSystem(session_id="comparison_session")
    embedder = await ModelManager.get_embedder()
    logger.info(f"문서 인덱싱 시작: {file_name}")
    await rag.load_document(file_path, file_name, embedder)
    
    # 2. 모델별 테스트 루프
    for model_name in models:
        logger.info(f"\n🚀 [MODEL] {model_name} 테스트 시작")
        
        for i, q in enumerate(questions):
            logger.info(f"[{i+1}/{len(questions)}] 질문: {q}")
            
            start_time = time.perf_counter()
            # RAG 쿼리 실행
            response_data = await rag.aquery(q, model_name=model_name)
            duration = time.perf_counter() - start_time
            
            # 메타데이터에서 페이지 정보 추출 (인용 확인용)
            docs = response_data.get("documents", [])
            pages = [str(d.metadata.get("page", "?")) for d in docs]
            
            results.append({
                "model": model_name,
                "question": q,
                "duration": duration,
                "pages": ", ".join(pages[:3]), # 상위 3개 페이지
                "response": response_data["response"][:100] + "..." # 답변 일부 저장
            })
            logger.info(f"완료 ({duration:.2f}s) | 인용 페이지: {pages[:3]}")
            
            # GPU VRAM 정리를 위한 짧은 대기
            await asyncio.sleep(1)

    # 3. 결과 요약 출력
    df = pd.DataFrame(results)
    summary = df.groupby("model")["duration"].agg(["mean", "min", "max"]).reset_index()
    
    print("\n" + "="*60)
    print("       QWEN 3 vs 3.5 (4B) BENCHMARK SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print("="*60)
    
    # 상세 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/compare_qwen_report_{timestamp}.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    logger.info(f"상세 보고서 저장됨: {report_path}")

if __name__ == "__main__":
    pdf_path = "tests/data/2201.07520v1.pdf" # CM3 논문 PDF
    test_questions = [
        "CM3 모델의 주요 특징 3가지는 무엇인가요?",
        "이 모델이 이미지를 학습할 때 사용하는 토큰화 방식은?",
        "DALL-E와 CM3의 차이점은 무엇인가요?",
        "논문의 저자들은 어떤 소속인가요?",
        "Casual Masking Objective란 무엇인가요?"
    ]
    
    if not os.path.exists(pdf_path):
        logger.error(f"테스트용 PDF 파일이 없습니다: {pdf_path}")
    else:
        asyncio.run(run_model_comparison(pdf_path, "2201.07520v1.pdf", test_questions))
