import asyncio
import os
import sys
import logging
import time

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.session import SessionManager
from core.model_loader import load_llm, load_embedding_model
from core.rag_core import build_rag_pipeline
from common.config import OLLAMA_MODEL_NAME, AVAILABLE_EMBEDDING_MODELS

# 로깅 레벨 조정 (테스트 시 가독성을 위해)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


async def test_full_rag_flow():
    print("\n" + "=" * 60)
    print("🚀 [통합 테스트] 실문서 기반 RAG 전체 플로우 검증")
    print("=" * 60)

    start_total = time.time()

    # 1. 세션 초기화
    SessionManager.init_session()

    # 2. 모델 로드
    model_name = OLLAMA_MODEL_NAME
    embed_model = AVAILABLE_EMBEDDING_MODELS[0]

    print(f"STEP 1: 모델 로딩 중... (LLM: {model_name})")
    llm = load_llm(model_name)
    embedder = load_embedding_model(embed_model)

    SessionManager.set("llm", llm)
    SessionManager.set("embedder", embedder)

    # 3. PDF 파일 경로 확인
    # tests/2201.07520v1.pdf 위치 확인
    pdf_path = os.path.join(os.path.dirname(__file__), "2201.07520v1.pdf")
    if not os.path.exists(pdf_path):
        # 현재 위치가 root인 경우 대응
        pdf_path = "tests/2201.07520v1.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ 오류: 테스트용 PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return

    print(
        f"STEP 2: 파이프라인 구축 및 인덱싱 시작 (파일: {os.path.basename(pdf_path)})"
    )

    # 4. RAG 파이프라인 구축
    indexing_start = time.time()
    success_msg, cache_used = build_rag_pipeline(
        uploaded_file_name="2201.07520v1.pdf", file_path=pdf_path, embedder=embedder
    )
    indexing_time = time.time() - indexing_start
    print(f"   - 완료: {success_msg}")
    print(f"   - 소요 시간: {indexing_time:.2f}초 (캐시 사용: {cache_used})")

    # 5. 질문 테스트
    qa_chain = SessionManager.get("qa_chain")
    if not qa_chain:
        print("❌ 오류: QA Chain이 생성되지 않았습니다.")
        return

    question = "이 논문에서 제안하는 핵심 기술인 'Causal Masking'이 무엇인지 한국어로 설명해줘."
    print("\nSTEP 3: 질문 생성 및 검색 실행")
    print(f"   - 질문: '{question}'")

    # LangGraph 실행 설정
    config = {"configurable": {"llm": llm}}

    print("STEP 4: 답변 생성 및 추론 중...")
    inference_start = time.time()

    try:
        # 실제 시스템과 동일하게 ainvoke 호출
        result = await qa_chain.ainvoke({"input": question}, config=config)
        full_response = result.get("response", "")
        docs = result.get("documents", [])

        inference_time = time.time() - inference_start

        print("\n" + "📜 [LLM 답변 수신]")
        print("-" * 60)
        if full_response:
            print(full_response)
        else:
            print("⚠️ 답변 내용이 비어 있습니다.")
        print("-" * 60)

        print("\n📊 통계:")
        print(f"   - 추론 소요 시간: {inference_time:.2f}초")
        print(f"   - 검색된 컨텍스트 수: {len(docs)}개")

        # 6. 최종 검증
        print("\nSTEP 5: 결과 검증")

        # 검증 포인트 1: 에러 메시지가 아님
        if "❌" in full_response or "오류" in full_response:
            print("   ❌ 검증 실패: 답변에 오류 메시지가 포함되어 있습니다.")
        # 검증 포인트 2: 컨텍스트 인용 포함 여부 ( [p. 로 시작하는 인용구 확인)
        elif "[p." in full_response:
            print("   ✅ 검증 성공: 문서 인용([p.X])이 포함된 정상적인 답변입니다.")
        # 검증 포인트 3: 최소 길이
        elif len(full_response) > 100:
            print("   ✅ 검증 성공: 충분한 길이의 답변이 생성되었습니다.")
        else:
            print("   ⚠️ 검증 주의: 답변이 생성되었으나 형식이 예상과 다를 수 있습니다.")

    except Exception as e:
        print(f"\n❌ 실행 중 치명적 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()

    total_time = time.time() - start_total
    print(f"\n🏁 전체 테스트 완료 (총 소요 시간: {total_time:.2f}초)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_full_rag_flow())
