import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.runnables import RunnableConfig

from common.config import OLLAMA_MODEL_NAME
from core.graph_builder import build_graph
from core.model_loader import load_llm


async def test_streaming_buffer():
    print(f"🚀 [Test] 스트리밍 버퍼링 검증 시작 (모델: {OLLAMA_MODEL_NAME})")

    # 1. 그래프 준비 (리트리버 없이 최소 기능으로 빌드)
    llm = load_llm(OLLAMA_MODEL_NAME)
    graph = build_graph()

    config = RunnableConfig(configurable={"llm": llm}, callbacks=[])

    inputs = {
        "input": "인공지능의 미래에 대해 3문장으로 짧게 설명해줘.",
        "context": "인공지능은 계속 발전하고 있습니다. 미래에는 더 많은 영역에서 활용될 것입니다.",
    }

    chunk_sizes = []
    total_events = 0
    full_response = ""

    print("--- [Streaming Monitoring] ---")

    # graph.astream_events를 사용하여 커스텀 이벤트 캡처
    async for event in graph.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event")

        # 우리가 정의한 'response_chunk' 이벤트 필터링
        if kind == "on_custom_event" and event.get("name") == "response_chunk":
            chunk_data = event.get("data", {})
            chunk_text = chunk_data.get("chunk", "")

            if chunk_text:
                total_events += 1
                chunk_sizes.append(len(chunk_text))
                full_response += chunk_text
                # 시각적으로 청크 단위를 확인하기 위해 구분자 표시
                print(f"[{len(chunk_text)}]", end="", flush=True)

    print("\n\n--- [Test Result] ---")
    if total_events > 0:
        avg_size = sum(chunk_sizes) / total_events
        print(f"📊 총 이벤트 횟수: {total_events}")
        print(f"📊 평균 청크 크기: {avg_size:.1f} 자")
        print(f"📊 청크 크기 분포: {chunk_sizes[:15]} ...")

        # 검증: 버퍼링이 작동한다면 대부분의 청크 크기가 1보다 커야 함
        one_char_chunks = [s for s in chunk_sizes if s <= 1]
        one_char_ratio = (len(one_char_chunks) / total_events) * 100

        if avg_size > 3:
            print(
                f"✅ 결과: 성공! (평균 청크 크기가 {avg_size:.1f}자로 버퍼링이 잘 작동하고 있습니다.)"
            )
        else:
            print(
                f"⚠️ 결과: 주의 (평균 청크 크기가 {avg_size:.1f}자로 낮습니다. 모델의 토큰 생성 단위가 클 수 있습니다.)"
            )
    else:
        print("❌ 이벤트를 수신하지 못했습니다.")


if __name__ == "__main__":
    asyncio.run(test_streaming_buffer())
