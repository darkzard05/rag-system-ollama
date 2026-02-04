import json
import time
import requests
import os
import asyncio
import httpx

async def measure_streaming_metrics():
    base_url = "http://127.0.0.1:8000/api/v1"
    api_key = os.getenv("TEST_API_KEY", "test-key-123")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print(f"Connecting to {base_url} with API Key: {api_key}")
    
    query = "대한민국의 수도는 어디인가요? 간결하게 답해주세요."
    payload = {"query": query, "session_id": "test_metrics_session"}
    
    # 1. 문서 업로드가 필요한지 확인 (생략 가능하면 생략, 필요하면 더미 업로드)
    # 여기서는 이미 업로드된 문서가 있다고 가정하거나, 에러를 통해 확인
    
    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    tokens = []
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{base_url}/stream_query", json=payload, headers=headers) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(await response.aread())
                    return

                print("Stream started...")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    receive_time = time.perf_counter()
                    
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "content" in data:
                                content = data["content"]
                                if content:
                                    if first_token_time is None:
                                        first_token_time = receive_time
                                        print(f"TTFT: {(first_token_time - start_time)*1000:.2f}ms")
                                    
                                    tokens.append(content)
                                    last_token_time = receive_time
                                    # print(content, end="", flush=True)
                        except:
                            pass
                    elif line.startswith("event: "):
                        event_type = line[7:]
                        # print(f"\n[{event_type}] ", end="")
                        if event_type == "end":
                            break
                            
    except Exception as e:
        print(f"\nError during streaming: {e}")
        return

    end_time = time.perf_counter()
    
    if not tokens:
        print("No tokens received.")
        return

    total_duration = end_time - start_time
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
    generation_time = (last_token_time - first_token_time) if first_token_time and last_token_time else 0
    tps = len("".join(tokens).split()) / generation_time if generation_time > 0 else 0
    
    print("\n" + "="*50)
    print(f"Streaming Metrics Summary:")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Time to First Token (TTFT): {ttft:.2f}ms")
    print(f"Generation Duration: {generation_time:.2f}s")
    print(f"Total Tokens (approx): {len("".join(tokens).split())}")
    print(f"Tokens Per Second (TPS): {tps:.2f}")
    print("="*50)

if __name__ == "__main__":
    # 이 스크립트는 서버가 실행 중이어야 합니다.
    asyncio.run(measure_streaming_metrics())
