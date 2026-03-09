"""
Qwen 3 (4B 모델) 전용 성능 및 품질 검증 벤치마크.
최적화(리랭커 경량화, 컨텍스트 축소) 적용 후의 실제 성능을 측정합니다.
"""

import os
import logging
import warnings

# --- Streamlit 관련 경고 차단 (임포트 전 실행 필수) ---
os.environ["STREAMLIT_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

import asyncio
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

async def run_qwen3_benchmark(file_path: str, file_name: str, questions: List[str]):
    """
    Qwen 3 모델에 대해 성능 지표를 수집합니다.
    """
    model_name = "qwen3:4b-instruct-2507-q4_K_M"
    results = []
    
    # 1. RAG 시스템 준비
    rag = RAGSystem(session_id="qwen3_test_session")
    embedder = await ModelManager.get_embedder()
    logger.info(f"🚀 Qwen 3 성능 검증 시작 (문서: {file_name})")
    
    start_idx = time.perf_counter()
    await rag.load_document(file_path, file_name, embedder)
    indexing_time = time.perf_counter() - start_idx
    logger.info(f"인덱싱 완료: {indexing_time:.2f}s")
    
    # 2. 테스트 루프
    for i, q in enumerate(questions):
        logger.info(f"\n[{i+1}/{len(questions)}] 질문: {q}")
        
        start_time = time.perf_counter()
        # RAG 쿼리 실행
        response_data = await rag.aquery(q, model_name=model_name)
        duration = time.perf_counter() - start_time
        
        # 메타데이터에서 페이지 정보 추출 (인용 확인용)
        docs = response_data.get("documents", [])
        pages = [str(d.metadata.get("page", "?")) for d in docs]
        
        results.append({
            "question": q,
            "duration": duration,
            "pages": ", ".join(pages[:3]),
            "response_summary": response_data["response"][:100].replace("\n", " ") + "..."
        })
        logger.info(f"✅ 응답 완료 ({duration:.2f}s) | 인용 페이지: {pages[:3]}")
        
        # 짧은 휴식 (자원 정리)
        await asyncio.sleep(0.5)

    # 3. 결과 출력
    df = pd.DataFrame(results)
    avg_time = df["duration"].mean()
    
    print("\n" + "="*60)
    print(f"       QWEN 3 PERFORMANCE REPORT (Optimized)")
    print("="*60)
    print(f"Indexing Time: {indexing_time:.2f}s")
    print(f"Avg Response Time: {avg_time:.2f}s")
    print("-" * 60)
    print(df[["question", "duration", "pages"]].to_string(index=False))
    print("="*60)
    
    # 상세 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/qwen3_optimized_report_{timestamp}.csv"
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
        asyncio.run(run_qwen3_benchmark(pdf_path, "2201.07520v1.pdf", test_questions))
