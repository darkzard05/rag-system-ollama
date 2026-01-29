import asyncio
import sys
import io
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm
from common.config import OLLAMA_MODEL_NAME, AVAILABLE_EMBEDDING_MODELS

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def test_real_app_condition_flow():
    print("🧪 [실제 앱 조건 테스트] 실제 파일 인덱싱부터 답변까지 전 과정 검증")

    # 1. 테스트용 실제 PDF 파일 생성 (PyMuPDF 활용)
    import fitz

    pdf_path = "tests/real_test_sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "RAG-System-Ollama is a local RAG implementation.")
    page.insert_text((50, 100), "It uses LangGraph for workflow orchestration.")
    page.insert_text(
        (50, 150), "The system supports DeepSeek-R1 and other Ollama models."
    )
    doc.save(pdf_path)
    doc.close()
    print(f"✅ 실제 테스트용 PDF 생성 완료: {pdf_path}")

    try:
        # 2. 실제 앱과 동일한 RAG 시스템 초기화
        # (임베딩 모델은 설정된 첫 번째 모델 사용)
        embedding_model = AVAILABLE_EMBEDDING_MODELS[0]
        rag_system = RAGSystem(embedding_model_name=embedding_model)

        # 3. 실제 인덱싱 (이 과정에서 청킹, 임베딩, 벡터 저장이 실제로 일어남)
        print("⚙️ 실제 인덱싱 시작 (청킹/임베딩)...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            # 실제 앱의 파일 처리 로직과 동일
            await rag_system.build_index(pdf_bytes, pdf_path)
        print("✅ 실제 인덱싱 및 벡터 DB 구축 완료")

        # 4. 실제 앱과 동일한 QA Chain 획득
        llm = load_llm(OLLAMA_MODEL_NAME)
        qa_chain = rag_system.get_qa_chain()
        run_config = {"configurable": {"llm": llm}}

        # 5. 실제 질문 던지기
        question = "이 시스템이 무엇을 사용하여 워크플로우를 관리하나요?"
        print(f"질문: {question}")

        full_response = ""
        async for event in qa_chain.astream_events(
            {"input": question}, config=run_config, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    full_response += content

        print("\n" + "=" * 50)
        print("실제 앱 엔진 답변:")
        print(full_response)
        print("=" * 50)

        # 6. 검증
        has_langgraph = "LangGraph" in full_response
        has_citation = "[p.1]" in full_response

        print("\n검증 결과:")
        print(
            f" - LangGraph 포함 (검색 성공 여부): {'✅ PASS' if has_langgraph else '❌ FAIL'}"
        )
        print(f" - 인용구 포함 ([p.1]): {'✅ PASS' if has_citation else '❌ FAIL'}")

    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


if __name__ == "__main__":
    asyncio.run(test_real_app_condition_flow())
