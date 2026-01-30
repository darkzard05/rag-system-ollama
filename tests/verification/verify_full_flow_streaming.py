import asyncio
import logging
import os
import sys
import time
from contextlib import aclosing

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from common.config import AVAILABLE_EMBEDDING_MODELS, OLLAMA_MODEL_NAME
from core.model_loader import load_embedding_model, load_llm
from core.rag_core import build_rag_pipeline
from core.session import SessionManager

# 로깅 레벨 조정 (핵심 로그만 확인)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_full_rag_flow_streaming():
    print("\n" + "=" * 65)
    print("🚀 [정밀 통합 테스트] 스트리밍 추적 기반 RAG 전체 플로우 검증")
    print("=" * 65)

    SessionManager.init_session()

    # 1. 모델 로드 (리랭커 자원 최적화 적용됨)
    model_name = OLLAMA_MODEL_NAME
    embed_model = AVAILABLE_EMBEDDING_MODELS[0]

    print(f"STEP 1: 모델 및 임베딩 준비... (LLM: {model_name})")
    llm = load_llm(model_name)
    embedder = load_embedding_model(embed_model)
    SessionManager.set("llm", llm)
    SessionManager.set("embedder", embedder)

    # 2. PDF 인덱싱 (기존 캐시 활용)
    pdf_path = os.path.join(os.path.dirname(__file__), "2201.07520v1.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = "tests/data/2201.07520v1.pdf"

    print("STEP 2: RAG 파이프라인 구축 (캐시 확인 중...)")
    build_rag_pipeline("2201.07520v1.pdf", pdf_path, embedder)

    # 3. 정밀 스트리밍 질문 테스트
    qa_chain = SessionManager.get("qa_chain")
    question = "이 논문에서 설명하는 'Causal Masking'의 동작 방식을 기술적인 관점에서 요약해줘."
    print("\nSTEP 3: 스트리밍 추론 시작")
    print(f"   - 질문: '{question}'")

    config = {"configurable": {"llm": llm}}
    full_response = ""
    token_count = 0
    start_inference = time.time()
    first_token_time = None

    try:
        # astream_events를 사용하여 내부 이벤트를 직접 추적
        async with aclosing(
            qa_chain.astream_events({"input": question}, config=config, version="v1")
        ) as event_stream:
            async for event in event_stream:
                kind = event["event"]
                name = event.get("name", "")

                # 검색 단계 감지
                if kind == "on_chain_start" and name == "retrieve":
                    print("   🔍 문맥 검색 중...")

                # 리랭킹 단계 감지
                elif kind == "on_chain_start" and name == "rerank_documents":
                    print("   🎯 리랭킹 수행 중 (CPU 최적화 적용)...")

                # 답변 생성 토큰 감지
                elif kind == "on_chat_model_stream":
                    content = event["data"].get("chunk", "")
                    if hasattr(content, "content"):
                        content = content.content

                    if content:
                        if first_token_time is None:
                            first_token_time = time.time() - start_inference
                            print(
                                f"   ✨ 첫 번째 토큰 수신 완료! (지연 시간: {first_token_time:.2f}초)"
                            )
                            print("   📝 답변 생성: ", end="", flush=True)

                        full_response += content
                        token_count += 1
                        print(content, end="", flush=True)

                # 커스텀 이벤트 (response_chunk) 감지
                elif kind == "on_custom_event" and name == "response_chunk":
                    chunk = event["data"].get("chunk", "")
                    if chunk and not first_token_time:  # 백업용
                        print(chunk, end="", flush=True)

        print("\n" + "-" * 65)
        inference_time = time.time() - start_inference

        # 4. 결과 분석 및 검증
        print("\n📊 [정밀 분석 보고서]")
        print(f"   - 총 추론 시간: {inference_time:.2f}초")
        print(
            f"   - 첫 토큰 지연(TTFT): {first_token_time if first_token_time else 'N/A'}"
        )
        print(f"   - 생성된 토큰 수: 약 {token_count}개")
        print(
            f"   - 평균 생성 속도: {token_count / inference_time:.2f} tokens/sec"
            if token_count > 0
            else "   - 생성 속도 측정 불가"
        )

        if len(full_response) > 50:
            print("\n✅ 최종 판정: 테스트 통과 (실질적인 답변 생성 확인됨)")
            if "[p." in full_response:
                print("   🔗 특이사항: 문서 인용 정보가 포함되어 답변의 근거가 명확함")
        else:
            print("\n❌ 최종 판정: 테스트 실패 (답변이 생성되지 않았거나 너무 짧음)")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(test_full_rag_flow_streaming())
