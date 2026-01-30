import asyncio
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.config import OLLAMA_MODEL_NAME
from core.model_loader import load_llm


async def debug_pure_streaming():
    print(f"🔍 모델 로딩: {OLLAMA_MODEL_NAME}")
    llm = load_llm(OLLAMA_MODEL_NAME)

    question = "Hello, how are you? Please give me a long response."
    print(f"🤔 질문: {question}")
    print("--- 스트리밍 시작 ---")

    start_time = asyncio.get_event_loop().time()
    first_token_received = False
    full_response = ""

    try:
        # 1. 모델 직접 스트리밍 테스트
        async for chunk in llm.astream(question):
            if not first_token_received:
                ttft = asyncio.get_event_loop().time() - start_time
                print(f"\n🚀 첫 토큰 수신! (TTFT: {ttft:.2f}s)")
                first_token_received = True

            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not content:
                print(f"\n[Empty Chunk Detection] {type(chunk)}: {chunk}")
            print(content, end="", flush=True)
            full_response += content

        print("\n\n--- 스트리밍 종료 ---")
        print(f"📊 최종 답변 길이: {len(full_response)}자")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    asyncio.run(debug_pure_streaming())
