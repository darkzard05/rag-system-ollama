import asyncio
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.runnables import RunnableConfig

from common.config import OLLAMA_MODEL_NAME
from core.graph_builder import build_graph
from core.model_loader import load_llm


async def test_prompt_caching():
    print("🚀 [Test] 프롬프트 캐싱(KV Cache) 최적화 검증 시작")

    llm = load_llm(OLLAMA_MODEL_NAME)
    graph = build_graph()

    config = RunnableConfig(configurable={"llm": llm}, callbacks=[])

    # 동일한 컨텍스트를 사용하도록 설정
    shared_context = [
        "인공지능(AI)은 인간의 지능을 기계로 구현한 것입니다.",
        "딥러닝은 인공지능의 한 분야로, 신경망을 통해 데이터를 학습합니다.",
        "RAG는 검색 증강 생성의 약자로, 외부 데이터를 통해 답변의 정확도를 높입니다.",
    ]

    async def run_query(query, label):
        print(f"\n--- [{label}] 실행: '{query}' ---")
        inputs = {
            "input": query,
            "documents": [
                # 검색 결과의 순서가 바뀐 상황을 가정하여 전달해도 정렬 로직이 해결해야 함
                {
                    "page_content": shared_context[2],
                    "metadata": {"source": "doc.pdf", "page": 3, "chunk_index": 2},
                },
                {
                    "page_content": shared_context[0],
                    "metadata": {"source": "doc.pdf", "page": 1, "chunk_index": 0},
                },
                {
                    "page_content": shared_context[1],
                    "metadata": {"source": "doc.pdf", "page": 2, "chunk_index": 1},
                },
            ],
        }

        start_time = time.time()
        ttft = 0

        # 실제 이벤트 스트림에서 첫 토큰 시간 측정
        async for event in graph.astream_events(inputs, config=config, version="v2"):
            if (
                event.get("event") == "on_custom_event"
                and event.get("name") == "response_chunk"
            ):
                if ttft == 0:
                    ttft = time.time() - start_time
                    print(f"⚡ 첫 토큰 도달 시간 (TTFT): {ttft:.2f}s")

        total_time = time.time() - start_time
        return ttft, total_time

    # 1회차: 캐시 미스 (최초 로딩)
    ttft1, total1 = await run_query("인공지능이 뭐야?", "1회차 (Cold Start)")

    # 2회차: 캐시 히트 (문서 순서가 바뀌어서 들어와도 정렬되어야 함)
    ttft2, total2 = await run_query("RAG에 대해 설명해줘.", "2회차 (Warm Start)")

    print("\n" + "=" * 40)
    print("📈 프롬프트 캐싱 결과 비교")
    print(f"  - 1회차 TTFT: {ttft1:.2f}s")
    print(f"  - 2회차 TTFT: {ttft2:.2f}s")

    if ttft2 < ttft1:
        improvement = (ttft1 - ttft2) / ttft1 * 100
        print(f"  - 개선율: {improvement:.1f}%")
        print("✅ 결과: 성공! 캐시 활용으로 인해 응답 속도가 대폭 향상되었습니다.")
    else:
        print("❌ 결과: 실패 또는 미미함. Ollama 설정이나 자원 상황을 확인해야 합니다.")
    print("=" * 40)


if __name__ == "__main__":
    # 테스트를 위해 Document 객체 Mocking 필요 (기존 코드가 Document 객체를 기대하므로)

    # 런타임에 graph_builder 내의 state 입력을 Document 객체로 변환하도록 패치하거나
    # 테스트 코드 내에서 Document 객체로 생성해서 전달
    asyncio.run(test_prompt_caching())
