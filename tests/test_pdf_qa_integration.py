import asyncio
import os
import sys
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from core.graph_builder import build_graph
from common.config import OLLAMA_MODEL_NAME, AVAILABLE_EMBEDDING_MODELS
from common.utils import apply_tooltips_to_response


@pytest.mark.asyncio
async def test_pdf_qa_full_graph():
    """
    실제 PDF 파일과 전체 LangGraph 파이프라인(build_graph)을 사용하여
    엔드투엔드 RAG 흐름을 검증합니다.
    """
    pdf_path = "tests/2201.07520v1.pdf"
    assert os.path.exists(pdf_path), f"테스트용 PDF 파일이 없습니다: {pdf_path}"

    session_id = "test_full_graph_session"
    embedding_model_name = AVAILABLE_EMBEDDING_MODELS[0]

    try:
        # 1. 모델 및 시스템 준비
        embedder = load_embedding_model(embedding_model_name)
        llm = load_llm(OLLAMA_MODEL_NAME)
        rag_system = RAGSystem(session_id=session_id)

        print("⚙️ 문서 로드 및 인덱싱 중...")
        await asyncio.to_thread(
            rag_system.load_document, pdf_path, "test.pdf", embedder
        )

        # 2. 전체 그래프 구축
        qa_chain = build_graph(retriever=rag_system.ensemble_retriever)
        run_config = {"configurable": {"llm": llm}}

        # 3. 질문 실행
        question = "CM3가 무엇인가요?"
        print(f"🤔 질문: {question}")

        full_response = ""
        retrieved_docs = []

        # astream_events를 사용하여 실제 앱과 동일한 스트리밍 로직 검증
        async for event in qa_chain.astream_events(
            {"input": question}, config=run_config, version="v2"
        ):
            kind = event["event"]
            name = event.get("name", "")

            # 1. 커스텀 토큰 이벤트 처리 (adispatch_custom_event)
            if kind == "on_custom_event" and name == "response_chunk":
                chunk_text = event["data"].get("chunk", "")
                if chunk_text:
                    if not full_response:
                        print("🚀 첫 번째 토큰 수신!")
                    full_response += chunk_text
                    # 실제 앱처럼 실시간으로 출력 (선택 사항)
                    # print(chunk_text, end="", flush=True)

            # 2. 상태 업데이트 이벤트 처리
            elif kind == "on_custom_event" and name == "status_update":
                status_msg = event["data"].get("message", "")
                print(f"📡 [Status] {status_msg}")

            # 3. 모델 스트림 (폴백 또는 직접 호출 대비)
            elif kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content and not full_response:  # 커스텀 이벤트가 없을 때만
                    full_response += content

            # 4. 문서 검색 완료 이벤트
            elif kind == "on_chain_end" and name == "retrieve":
                retrieved_docs = event["data"]["output"]["documents"]
                print(f"📚 문서 {len(retrieved_docs)}개 검색 완료")

        # 4. 결과 검증
        if not full_response:
            print("⚠️ 스트리밍 실패, ainvoke 폴백 시도...")
            result = await qa_chain.ainvoke({"input": question}, config=run_config)
            full_response = result.get("response", "")
            retrieved_docs = result.get("documents", [])

        print(f"🤖 최종 답변 길이: {len(full_response)}자")

        # 5. UI 포맷팅 적용
        final_content = apply_tooltips_to_response(full_response, retrieved_docs)

        assert len(full_response) > 0, "답변이 생성되지 않았습니다."
        assert len(retrieved_docs) > 0, "문서가 검색되지 않았습니다."

        print("\n--- 최종 답변 샘플 ---")
        print(final_content[:500] + "...")

    except Exception as e:
        pytest.fail(f"전체 그래프 테스트 실패: {e}")


if __name__ == "__main__":
    asyncio.run(test_pdf_qa_full_graph())
