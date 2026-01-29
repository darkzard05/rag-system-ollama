import asyncio
import sys
import io
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from core.graph_builder import build_graph
from common.config import OLLAMA_MODEL_NAME, AVAILABLE_EMBEDDING_MODELS
from common.utils import apply_tooltips_to_response

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def test_with_real_user_pdf():
    pdf_path = "tests/2201.07520v1.pdf"
    print(f"🚀 [실제 사용자 PDF 테스트] 파일: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        return

    try:
        # 1. 실제 임베딩 모델 로드
        embedding_model_name = AVAILABLE_EMBEDDING_MODELS[0]
        print(f"⚙️ 임베딩 모델 로드 중: {embedding_model_name}")
        embedder = load_embedding_model(embedding_model_name)

        # 2. RAG 시스템 초기화 및 인덱싱
        rag_system = RAGSystem(session_id="test_session_real")

        print("⚙️ 문서 분석 및 인덱싱 시작 (PyMuPDF + FAISS)...")
        status_msg, success = await asyncio.to_thread(
            rag_system.load_document, pdf_path, "2201.07520v1.pdf", embedder
        )

        if not success:
            print(f"❌ 인덱싱 실패: {status_msg}")
            return
        print(f"✅ 인덱싱 결과: {status_msg}")

        # 3. 실제 앱과 동일한 그래프(체인) 생성
        # RAGSystem 내부에서 생성된 ensemble_retriever를 사용합니다.
        print("⚙️ QA 그래프 구성 중...")
        qa_chain = build_graph(retriever=rag_system.ensemble_retriever)

        # 4. LLM 및 실행 설정
        llm = load_llm(OLLAMA_MODEL_NAME)
        run_config = {"configurable": {"llm": llm}}

        # 5. 실제 질문 던지기
        # 논문 주제인 Chain of Thought (CoT)에 대해 질문
        question = (
            "논문에서 설명하는 Chain of Thought 프롬프팅의 핵심 아이디어가 무엇인가요?"
        )
        print(f"\n질문: {question}")
        print("답변 생성 중 (스트리밍).", end="", flush=True)

        full_response = ""
        retrieved_docs = []

        # 실제 앱의 스트리밍 및 이벤트 수신 로직 재현
        async for event in qa_chain.astream_events(
            {"input": question}, config=run_config, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    full_response += content
                    if len(full_response) % 10 == 0:
                        print(".", end="", flush=True)
            elif kind == "on_chain_end" and event["name"] == "retrieve":
                # 실제 검색된 문서 조각들 수집
                retrieved_docs = event["data"]["output"]["documents"]

        print(f"\n✅ 답변 생성 시도 완료 (응답 길이: {len(full_response)}자)")

        # [추가] 검색된 문서 조각 직접 확인
        print("\n🔍 [검색 품질 확인] 실제로 문서에서 찾은 관련 문장들 (상위 3개):")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            print(f"--- Document {i} (p.{doc.metadata.get('page')}) ---")
            print(doc.page_content[:200] + "...")

        # 6. UI 포맷팅 (툴팁 변환) 적용
        final_ui_content = apply_tooltips_to_response(full_response, retrieved_docs)

        # 7. 결과 출력 및 분석
        print("\n" + "=" * 50)
        print("📋 실제 앱 엔진 최종 결과 (샘플):")
        print(final_ui_content[:800] + "...")
        print("=" * 50)

        # 검증
        has_citations = "[p." in final_ui_content
        has_tooltips = 'class="tooltip"' in final_ui_content
        has_logic = len(retrieved_docs) > 0

        print("\n검증 결과:")
        print(f" - 실제 문서 검색 성공: {'✅ PASS' if has_logic else '❌ FAIL'}")
        print(
            f" - 인용구 및 툴팁 변환: {'✅ PASS' if has_citations and has_tooltips else '❌ FAIL'}"
        )

    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_with_real_user_pdf())
