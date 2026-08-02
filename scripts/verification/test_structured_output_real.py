import asyncio
import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from core.graph_builder import UnifiedGradeRewriteResponse


def _parse_unified_json(content: Any) -> dict:
    """LLM 출력에서 통합 응답 JSON을 추출합니다. (JSON 모드 + 수동 파싱 폴백)"""
    if not isinstance(content, str):
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        else:
            content = str(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def test_real_structured_output():
    # 1. 모델 초기화
    llm = ChatOllama(model="qwen3.5:4b", temperature=0)

    # 2. 통합 Grade/Rewrite 응답 테스트
    #    (새 API: with_structured_output 대신 bind(response_format=json) + JSON 파싱)
    print("\n--- Testing Unified Grade/Rewrite Response ---")
    json_llm = llm.bind(response_format={"type": "json_object"})

    grade_prompt = """사용자의 질문에 대해 아래 문서가 답변을 제공하기 위한 실질적인 근거를 포함하고 있는지 판단하세요.
질문: DeepSeek-R1의 성능은 어때?
문서:
DOC 1: DeepSeek-R1은 오픈 소스 모델 중 가장 뛰어난 추론 성능을 보이며, 특히 수학과 코딩 분야에서 GPT-4o에 필적하는 결과를 보여줍니다."""

    try:
        raw = await json_llm.ainvoke(grade_prompt)
        content = raw.content if hasattr(raw, "content") else str(raw)
        parsed = UnifiedGradeRewriteResponse(**_parse_unified_json(content))
        print(f"Is Relevant: {parsed.is_relevant}")
        print(f"Reason: {parsed.reason}")
    except Exception as e:
        print(f"Unified Response Error: {e}")

    # 3. Rewrite 테스트 (action="rewrite" + optimized_query)
    print("\n--- Testing Rewrite (Unified Response) ---")

    rewrite_prompt = """사용자의 질문을 분석하여 문서 저장소(RAG)에서 더 정확한 정보를 찾을 수 있도록 구체적인 검색어로 재구성하세요.
불필요한 수식어는 제거하고, 핵심 키워드와 맥락 위주로 작성하세요.
원본 질문: 딥식 R1이랑 GPT4 비교해줘"""

    try:
        raw = await json_llm.ainvoke(rewrite_prompt)
        content = raw.content if hasattr(raw, "content") else str(raw)
        parsed = UnifiedGradeRewriteResponse(**_parse_unified_json(content))
        print(f"Optimized Query: {parsed.optimized_query}")
    except Exception as e:
        print(f"Rewrite Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_real_structured_output())
