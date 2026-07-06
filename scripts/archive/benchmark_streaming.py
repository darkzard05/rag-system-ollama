import asyncio
import time
import json
import statistics
import httpx
import os
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class StreamMetrics:
    query: str
    ttft_ms: float
    tps: float
    itl_avg_ms: float
    itl_std_ms: float
    total_time_s: float
    total_tokens: int


async def benchmark_query(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    api_key: str,
    session_id: str = "benchmark",
):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"query": query, "session_id": session_id}

    start_time = time.perf_counter()
    first_token_time = None
    token_timestamps = []
    total_tokens = 0

    try:
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=120.0
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return None

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                current_time = time.perf_counter()
                data_str = line[6:].strip()

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # We only care about "message" events for TPS/ITL
                if data.get("type") == "message" or "content" in data:
                    if first_token_time is None:
                        first_token_time = current_time

                    token_timestamps.append(current_time)
                    # Rough token estimate: using character count / 4 for mixed lang,
                    # or we can just count chunks as 'units' if we want raw streaming speed.
                    # For a better TPS, we'd need actual tokens, but here we track chunk arrival.
                    total_tokens += 1

    except Exception as e:
        print(f"Request failed: {e}")
        return None

    end_time = time.perf_counter()

    if first_token_time is None:
        return None

    ttft = (first_token_time - start_time) * 1000
    total_time = end_time - start_time

    # ITL (Inter-Token Latency)
    itls = []
    for i in range(1, len(token_timestamps)):
        itls.append((token_timestamps[i] - token_timestamps[i - 1]) * 1000)

    itl_avg = statistics.mean(itls) if itls else 0
    itl_std = statistics.stdev(itls) if len(itls) > 1 else 0
    tps = total_tokens / total_time if total_time > 0 else 0

    return StreamMetrics(
        query=query,
        ttft_ms=ttft,
        tps=tps,
        itl_avg_ms=itl_avg,
        itl_std_ms=itl_std,
        total_time_s=total_time,
        total_tokens=total_tokens,
    )


async def main():
    # Configuration
    API_URL = "http://127.0.0.1:8000/api/v1/stream_query"
    API_KEY = os.getenv("TEST_API_KEY", "benchmark_key")

    test_queries = [
        "이 문서의 주요 내용을 요약해줘",
        "시스템의 아키텍처에 대해 설명해줘",
        "최적화 방법론에 대해 상세히 알려줘",
        "짧게 인사해줘",
        "아주 긴 답변이 필요한 복잡한 질문을 해보겠습니다. 이 시스템의 전체적인 동작 과정과 각 컴포넌트의 역할, 그리고 데이터 흐름을 상세하게 설명해주세요.",
    ]

    print(f"Starting benchmark with {len(test_queries)} queries...")
    print(f"Target URL: {API_URL}")
    print(f"API Key: {API_KEY}")
    print("-" * 60)

    results = []
    async with httpx.AsyncClient() as client:
        for i, query in enumerate(test_queries):
            print(
                f"Query {i + 1}/{len(test_queries)}: {query[:30]}...",
                end=" ",
                flush=True,
            )
            metric = await benchmark_query(client, API_URL, query, API_KEY)
            if metric:
                print(f"Done (TTFT: {metric.ttft_ms:.2f}ms, TPS: {metric.tps:.2f})")
                results.append(metric)
            else:
                print("Failed")

    if not results:
        print("No successful results.")
        return

    # Calculate Aggregates
    avg_ttft = statistics.mean([r.ttft_ms for r in results])
    avg_tps = statistics.mean([r.tps for r in results])
    avg_itl = statistics.mean([r.itl_avg_ms for r in results])
    avg_itl_std = statistics.mean([r.itl_std_ms for r in results])

    print("-" * 60)
    print("FINAL BENCHMARK RESULTS")
    print("-" * 60)
    print(f"Average TTFT: {avg_ttft:.2f} ms")
    print(f"Average TPS:  {avg_tps:.2f} tokens/s")
    print(f"Average ITL:  {avg_itl:.2f} ms")
    print(f"Average ITL StdDev: {avg_itl_std:.2f} ms")
    print("-" * 60)

    # Save to file
    with open("scripts/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print("Results saved to scripts/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
