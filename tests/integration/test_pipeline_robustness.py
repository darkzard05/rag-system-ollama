import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from common.config import DEFAULT_OLLAMA_MODEL
from core.graph_builder import build_graph
from core.model_loader import load_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobustnessTest")


async def test_pipeline_paths():
    llm = load_llm(DEFAULT_OLLAMA_MODEL)
    # 리트리버 없이도 작동하도록 빈 리트리버 주입
    app = build_graph(retriever=None)

    config = {"configurable": {"llm": llm}}

    print("\n=== 파이프라인 경로별 데이터 무결성 테스트 시작 ===")

    # 1. 인사말 경로 (GREETING)
    print("\n[*] Case 1: 인사말 (예상: context 초기화 및 친절한 응답)")
    inputs = {"input": "안녕? 너는 누구니?"}
    result = await app.ainvoke(inputs, config=config)

    print(f"의도 결정: {result.get('route_decision')}")
    print(f"컨텍스트 상태: '{result.get('context')[:50]}'...")
    print(f"최종 답변: {result.get('response')}")

    if result.get("route_decision") == "GREETING" and len(result.get("response")) > 0:
        print("✅ GREETING 경로 무결성 확인")
    else:
        print("❌ GREETING 경로 데이터 오류")

    # 2. 단순 사실 경로 (FACTOID)
    print("\n[*] Case 2: 단순 사실 질문 (예상: 확장 생략, 원본 질문 사용)")
    inputs = {"input": "이 문서의 제목이 뭐야?"}
    result = await app.ainvoke(inputs, config=config)

    print(f"의도 결정: {result.get('route_decision')}")
    print(f"검색 쿼리: {result.get('search_queries')}")

    if result.get("route_decision") == "FACTOID" and result.get("search_queries") == [
        inputs["input"]
    ]:
        print("✅ FACTOID 경로 무결성 확인")
    else:
        print("❌ FACTOID 경로 데이터 오류")


if __name__ == "__main__":
    asyncio.run(test_pipeline_paths())
