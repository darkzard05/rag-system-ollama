import asyncio
import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from common.config import OLLAMA_MODEL_NAME
from core.model_loader import load_llm


async def test_qwen_reasoning_flow():
    print("\n" + "=" * 60)
    print(f"🧠 [사고 과정 검증] 모델: {OLLAMA_MODEL_NAME}")
    print("=" * 60)

    # 1. 모델 로드
    llm = load_llm(OLLAMA_MODEL_NAME)

    # 2. 사고를 유도하는 복잡한 질문
    question = "방 안에 30명의 사람이 있고, 각각 서로 한 번씩 악수를 한다면 총 몇 번의 악수가 일어날까? 단계별로 생각해서 답해줘."

    print(f"질문: {question}\n")
    print("--- 스트리밍 분석 시작 ---")

    full_content = ""
    full_thought = ""

    try:
        # astream_events를 사용하여 모든 내부 데이터 구조 확인
        async for event in llm.astream_events(question, version="v1"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]

                # 1. 일반 콘텐츠 확인
                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get("content", "")

                if content:
                    full_content += content
                    print(f"[Content] {content}", end="", flush=True)

                # 2. 사고 과정(Thought/Reasoning) 확인
                # LangChain의 Ollama integration은 종종 additional_kwargs에 이를 담습니다.
                thought = ""
                if hasattr(chunk, "additional_kwargs"):
                    thought = chunk.additional_kwargs.get("thought", "")

                if thought:
                    if not full_thought:
                        print("\n\n[💡 사고 과정 감지됨!]")
                    full_thought += thought
                    print(f"{thought}", end="", flush=True)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

    print("\n\n" + "=" * 60)
    print("📊 최종 분석 결과:")
    print(f"- 최종 답변 길이: {len(full_content)}자")
    print(
        f"- 사고 과정 추출 여부: {'✅ 감지됨' if full_thought else '❌ 감지되지 않음 (또는 일반 텍스트에 포함됨)'}"
    )
    if full_thought:
        print(f"- 사고 과정 길이: {len(full_thought)}자")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_qwen_reasoning_flow())
