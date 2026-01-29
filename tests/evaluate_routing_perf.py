import asyncio
import time
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.model_loader import load_llm
from core.query_optimizer import RAGQueryOptimizer
from common.config import DEFAULT_OLLAMA_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Eval")


async def evaluate_routing():
    llm = load_llm(DEFAULT_OLLAMA_MODEL)

    test_cases = [
        {"q": "안녕하세요 반가워요!", "expected": "GREETING"},
        {"q": "이 문서의 작성자가 누구인가요?", "expected": "FACTOID"},
        {"q": "전체 내용을 요약하고 특징을 3가지만 분석해줘.", "expected": "RESEARCH"},
    ]

    print("\n=== 시맨틱 라우팅 성능 및 정확도 평가 ===")

    total_latency = 0
    for case in test_cases:
        start = time.time()
        result = await RAGQueryOptimizer.classify_intent(case["q"], llm)
        latency = (time.time() - start) * 1000
        total_latency += latency

        status = "✅" if result == case["expected"] else "❌"
        print(f"질문: {case['q'][:30]:<30}")
        print(f"결과: {result:<10} | 지연시간: {latency:7.2f}ms | 정확도: {status}")

    avg_latency = total_latency / len(test_cases)
    print(f"\n평균 라우팅 지연 시간: {avg_latency:.2f}ms")

    print("\n[비교 분석]")
    print(f"1. 라우팅 비용: 약 {avg_latency:.2f}ms")
    print("2. 쿼리 확장 비용(생략 시 이득): 약 2500ms ~ 4000ms")
    print(f"3. 단순 질문 시 예상 속도 향상: 약 {(3000 - avg_latency) / 1000:.2f}초")


if __name__ == "__main__":
    asyncio.run(evaluate_routing())
