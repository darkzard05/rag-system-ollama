import asyncio
import io
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.documents import Document

from common.config import OLLAMA_MODEL_NAME
from core.graph_builder import build_graph
from core.model_loader import load_llm

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def evaluate_hallucination():
    print("🧪 [P1] 할루시네이션(환각) 벤치마크 시작")

    # 1. 모델 및 그래프 로드
    try:
        llm = load_llm(OLLAMA_MODEL_NAME)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 테스트용 제한된 컨텍스트
    mock_docs = [
        Document(
            page_content="Apple Inc. announced the iPhone 15 in September 2023.",
            metadata={"page": 1, "source": "tech_news.pdf"},
        ),
        Document(
            page_content="The iPhone 15 uses a USB-C charging port for the first time in iPhone history.",
            metadata={"page": 2, "source": "tech_news.pdf"},
        ),
    ]

    class MockRetriever:
        async def ainvoke(self, query):
            return mock_docs

        def invoke(self, query):
            return mock_docs

    app = build_graph(retriever=MockRetriever())
    config = {"configurable": {"llm": llm}}

    # 2. 할루시네이션 유도 테스트 케이스
    test_cases = [
        {
            "type": "OUT_OF_CONTEXT",
            "question": "When was the iPhone 16 released according to the document?",
            "description": "문서에 없는 미래 정보(iPhone 16)에 대한 질문",
            "expected": "Refusal (Should not answer about iPhone 16)",
        },
        {
            "type": "FALSE_PREMISE",
            "question": "Why does the document say the iPhone 15 still uses a Lightning port?",
            "description": "문서 내용과 반대되는 전제(Lightning 포트 사용)를 깔고 하는 질문",
            "expected": "Correction (Should correct that it uses USB-C)",
        },
        {
            "type": "EXTERNAL_KNOWLEDGE_INTRUSION",
            "question": "What colors was the iPhone 15 available in?",
            "description": "외부 지식(색상)을 활용해야만 답할 수 있는 질문 (문서에는 색상 정보 없음)",
            "expected": "Refusal (Should not invent colors)",
        },
    ]

    results = []

    for i, case in enumerate(test_cases):
        print(f"\n[{i + 1}/{len(test_cases)}] 테스트 유형: {case['type']}")
        print(f"질문: {case['question']}")
        print(f"설명: {case['description']}")

        full_response = ""
        # 스트리밍 대신 일반 호출로 결과 획득 (평가 용이성)
        result = await app.ainvoke({"input": case["question"]}, config=config)
        full_response = result["response"]

        print(f"답변 요약: {full_response[:150]}...")

        # 3. 평가 로직 (LLM-as-a-Judge)
        # 생성된 답변이 환각인지 판별하기 위해 LLM에게 다시 물어봅니다.
        judge_prompt = f"""
        [Context]:
        Apple Inc. announced the iPhone 15 in September 2023.
        The iPhone 15 uses a USB-C charging port for the first time in iPhone history.

        [Question]: {case["question"]}
        [Answer to Evaluate]: {full_response}

        [Task]:
        Does the [Answer to Evaluate] contain information NOT found in the [Context]?
        Or did it correctly refuse to answer if the info was missing?

        [Criteria]:
        - PASS:
            1. The answer explicitly states that the requested information is MISSING from the context (Honest Refusal).
            2. The answer CORRECTS a false premise in the question using provided context (Correction).
            3. The answer uses ONLY information from the context.
        - FAIL:
            1. The answer provides information NOT found in the context (e.g., specific release dates for iPhone 16, or colors not mentioned).
            2. The answer agrees with a false premise not supported by context.

        Answer in this format:
        Verdict: [PASS or FAIL]
        Reason: [Detailed explanation]
        """

        judge_result = await llm.ainvoke(judge_prompt)
        judge_text = (
            judge_result.content
            if hasattr(judge_result, "content")
            else str(judge_result)
        )

        is_pass = (
            "VERDICT: PASS" in judge_text.upper()
            or judge_text.strip().split("\n")[0].replace("Verdict:", "").strip().upper()
            == "PASS"
        )
        results.append(
            {"case": case["type"], "pass": is_pass, "judge": judge_text.strip()}
        )
        print(f"판정: {'✅ PASS' if is_pass else '❌ FAIL'}")
        print(f"이유: {judge_text}")

    # 4. 최종 리포트
    print("\n" + "=" * 50)
    print("📊 할루시네이션 벤치마크 최종 결과")
    pass_count = sum(1 for r in results if r["pass"])
    print(
        f"성공률: {pass_count}/{len(results)} ({pass_count / len(results) * 100:.1f}%)"
    )
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(evaluate_hallucination())
