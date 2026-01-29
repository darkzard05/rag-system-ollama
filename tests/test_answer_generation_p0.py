import asyncio
import sys
import io
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.config import OLLAMA_MODEL_NAME
from core.model_loader import load_llm
from core.graph_builder import build_graph
from langchain_core.documents import Document

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def verify_language(text: str, expected_lang: str) -> bool:
    """언어 판별 (한글 글자 비율 기반)"""
    if not text:
        return False
    hangul_chars = [char for char in text if "\\\\uac00" <= char <= "\\\\ud7a3"]
    hangul_ratio = len(hangul_chars) / len(text)

    if expected_lang == "ko":
        return hangul_ratio > 0.05  # 한글이 5% 이상이면 한국어로 간주
    else:
        return hangul_ratio < 0.01  # 한글이 1% 미만이면 영어(또는 기타)로 간주


async def test_answer_generation_p0():
    print("🚀 [P0] 답변 생성 및 스트리밍 종합 테스트 시작")

    # 1. 환경 준비
    try:
        llm = load_llm(OLLAMA_MODEL_NAME)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    mock_docs = [
        Document(
            page_content="The capital of France is Paris.",
            metadata={"page": 1, "source": "geo.pdf"},
        ),
        Document(
            page_content="한국의 수도는 서울입니다.",
            metadata={"page": 2, "source": "geo.pdf"},
        ),
    ]

    class MockRetriever:
        async def ainvoke(self, query):
            return mock_docs

        def invoke(self, query):
            return mock_docs

    app = build_graph(retriever=MockRetriever())
    config = {"configurable": {"llm": llm}}

    # 2. 테스트 케이스 정의
    test_cases = [
        {
            "name": "언어 일관성 (한국어)",
            "input": "한국의 수도는 어디인가요?",
            "expected_lang": "ko",
            "must_include": ["서울", "[p.2]"],
        },
        {
            "name": "언어 일관성 (영어)",
            "input": "What is the capital of France?",
            "expected_lang": "en",
            "must_include": ["Paris", "[p.1]"],
        },
        {
            "name": "할루시네이션 방지 (정보 없음)",
            "input": "일본의 수도는 어디인가요?",
            "check_groundedness": True,
        },
    ]

    for case in test_cases:
        print(f"\n--- 테스트 항목: {case['name']} ---")
        print(f"질문: {case['input']}")

        full_response = ""
        events_count = 0
        metadata_received = False

        # 3. 스트리밍 이벤트 검증
        async for event in app.astream_events(
            {"input": case["input"]}, config=config, version="v2"
        ):
            kind = event["event"]

            # 메타데이터 이벤트 확인 (사용자 정의 이벤트)
            if kind == "on_custom_event" and event["name"] == "metadata_update":
                metadata_received = True

            # 채팅 모델 스트림 확인
            elif kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    full_response += content
                    events_count += 1
                    if events_count % 10 == 0:
                        print(".", end="", flush=True)

        print(f"\n응답 완료 ({events_count} chunks)")
        print("-" * 30)
        print(full_response)
        print("-" * 30)

        # 4. 검증 로직
        results = []

        # 언어 검증
        if "expected_lang" in case:
            lang_ok = await verify_language(full_response, case["expected_lang"])
            results.append(("언어 일치", lang_ok))

        # 필수 포함 단어 및 인용 검증
        if "must_include" in case:
            include_ok = all(word in full_response for word in case["must_include"])
            results.append(("필수 내용 포함", include_ok))

        # 할루시네이션 검증
        if case.get("check_groundedness"):
            is_honest = "도쿄" not in full_response or any(
                x in full_response for x in ["정보가", "알 수", "제공된"]
            )
            results.append(("할루시네이션 방지", is_honest))

        # 최종 리포트
        all_passed = True
        for label, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f" - {label}: {status}")
            if not passed:
                all_passed = False

        if not all_passed:
            print(f"⚠️ {case['name']} 테스트 실패")


if __name__ == "__main__":
    asyncio.run(test_answer_generation_p0())
