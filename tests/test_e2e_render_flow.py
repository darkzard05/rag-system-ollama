import asyncio
import sys
import io
import time
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.config import OLLAMA_MODEL_NAME
from core.model_loader import load_llm
from core.graph_builder import build_graph
from common.utils import apply_tooltips_to_response
from langchain_core.documents import Document

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def test_e2e_generation_to_display_flow():
    print("🚀 [E2E 통합 테스트] 생성 -> 포맷팅 -> UI 표시 흐름 검증 시작")

    # 1. 시스템 준비 (LLM 및 RAG 그래프)
    try:
        llm = load_llm(OLLAMA_MODEL_NAME)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 테스트용 문서 (인용구 테스트용)
    mock_docs = [
        Document(
            page_content="DeepSeek-R1 is a powerful reasoning model.",
            metadata={"page": 1, "source": "tech.pdf"},
        ),
        Document(
            page_content="It supports various local RAG implementations.",
            metadata={"page": 2, "source": "tech.pdf"},
        ),
    ]

    class MockRetriever:
        async def ainvoke(self, query):
            return mock_docs

        def invoke(self, query):
            return mock_docs

    app = build_graph(retriever=MockRetriever())
    config = {"configurable": {"llm": llm}}

    # 2. 질문 입력 및 답변 생성 (Generation)
    question = "DeepSeek-R1의 특징과 지원 사항을 요약해줘."
    print(f"질문: {question}")

    full_response = ""
    start_time = time.time()

    # 스트리밍 시뮬레이션
    async for event in app.astream_events(
        {"input": question}, config=config, version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                full_response += content
                if len(full_response) % 20 == 0:  # 진행 표시
                    print(".", end="", flush=True)

    print(f"\n✅ 답변 생성 완료 (소요시간: {time.time() - start_time:.2f}s)")

    # 3. 인용구 툴팁 포맷팅 (Formatting)
    # 실제 UI에서 _stream_chat_response가 호출하는 apply_tooltips_to_response 실행
    formatted_response = apply_tooltips_to_response(full_response, mock_docs)

    # 4. 최종 결과 검증 (Display Analysis)
    print("\n--- 최종 렌더링 결과 분석 ---")

    checks = {
        "구조화된 보고서 형식 (헤더 확인)": "# " in formatted_response
        and "## " in formatted_response,
        "인용구 툴팁 변환 (HTML 확인)": 'class="tooltip"' in formatted_response,
        "페이지 정보 포함 ([p.1] 확인)": "[p.1]" in formatted_response
        or "[p.2]" in formatted_response,
        "내용 정확성 (DeepSeek-R1 포함)": "DeepSeek-R1" in formatted_response,
    }

    all_passed = True
    for label, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f" - {label}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 모든 생성 및 UI 표시 흐름 테스트를 통과했습니다!")
        print("-" * 50)
        print("최종 출력물 샘플 (상위 300자):")
        print(formatted_response[:300] + "...")
        print("-" * 50)
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 결과물을 확인하세요.")
        print(formatted_response)


if __name__ == "__main__":
    asyncio.run(test_e2e_generation_to_display_flow())
