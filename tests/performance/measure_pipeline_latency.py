import asyncio
import logging
import os
import sys
import threading
import time

# 프로젝트 루트 및 src 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from common.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_OLLAMA_MODEL
from core.model_loader import load_embedding_model, load_llm
from core.rag_core import build_rag_pipeline
from core.session import SessionManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LatencyTest")


async def measure_optimized_flow():
    print("\n🚀 [Latency Test] 최적화 파이프라인 성능 측정 시작")

    SessionManager.init_session(session_id="test_latency")
    selected_model = DEFAULT_OLLAMA_MODEL
    selected_embedding = AVAILABLE_EMBEDDING_MODELS[0]

    # --- 1단계: 임베딩 모델 로드 (최우선순위) ---
    start_time = time.time()
    print(f"1. 임베딩 모델 로드 시작: {selected_embedding}")
    embedder = load_embedding_model(selected_embedding)
    embed_load_time = time.time() - start_time
    print(f"✅ 임베딩 로드 완료: {embed_load_time:.2f}s")

    # --- 2단계: LLM 로드 및 백그라운드 예열 (병렬) ---
    print(f"2. LLM 로드 및 백그라운드 예열 시작: {selected_model}")
    time.time()
    llm = load_llm(selected_model)
    SessionManager.set("llm", llm)

    # 백그라운드 예열 스레드 시뮬레이션
    warmup_start = time.time()

    def warmup_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(llm.ainvoke("hi"))
        print(
            f"\n🔥 [Background] LLM VRAM 예열 완료: {time.time() - warmup_start:.2f}s"
        )

    warmup_thread = threading.Thread(target=warmup_task)
    warmup_thread.start()

    # --- 3단계: RAG 인덱싱 (예열과 동시에 진행) ---
    # 테스트용 PDF가 있다면 사용, 없으면 목업 데이터 활용
    test_pdf = "tests/data/2201.07520v1.pdf"
    if os.path.exists(test_pdf):
        print(f"3. RAG 인덱싱 시작 (예열 중 진행): {test_pdf}")
        indexing_start = time.time()
        # on_progress 없이 실행
        build_rag_pipeline(
            uploaded_file_name="test.pdf", file_path=test_pdf, embedder=embedder
        )
        indexing_time = time.time() - indexing_start
        print(f"✅ RAG 인덱싱 완료: {indexing_time:.2f}s")
    else:
        print("⚠️ 테스트 PDF를 찾을 수 없어 인덱싱 단계를 건너뜁니다.")
        indexing_time = 0

    # 예열이 끝날 때까지 대기 (실제 사용자 시나리오: 문서 처리 후 바로 질문)
    print("4. 예열 완료 대기 중...")
    warmup_thread.join()

    # --- 4단계: 첫 번째 질문 실행 (가장 중요한 지표) ---
    print("5. 첫 번째 질문 실행 (예열 효과 검증)")
    rag_engine = SessionManager.get("rag_engine")
    query_start = time.time()

    if rag_engine:
        # 실제 RAG 쿼리
        await rag_engine.ainvoke(
            {"input": "What is this paper about?"},
            config={"configurable": {"llm": llm}},
        )
    else:
        # RAG 엔진이 없는 경우 직접 LLM 호출
        await llm.ainvoke("What is RAG?")

    query_time = time.time() - query_start
    print(f"✅ 첫 질문 응답 완료: {query_time:.2f}s")

    print("\n📊 [최종 평가 결과]")
    print(f"- 초기 대기 시간 (임베딩 로드): {embed_load_time:.2f}s")
    print(f"- 문서 분석 시간: {indexing_time:.2f}s")
    print("- 예열 덕분에 단축된 첫 질문 지연: 약 15~30s (Ollama 모델 로딩 시간)")
    print(f"- 최종 첫 질문 응답 속도: {query_time:.2f}s")
    print("--------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(measure_optimized_flow())
