import asyncio
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
# 실제 환경에서는 유효한 API 키가 필요합니다.
# api_server.py에서 생성된 키를 사용하거나 테스트용 키를 설정해야 합니다.
API_KEY = (
    "admin123"  # 기본적으로 생성되는 키 중 하나를 가정하거나 서버 로그에서 확인 필요
)


async def simulate_streaming_query(user_id: int, query: str, session_id: str):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"query": query, "session_id": session_id}

    print(f"[User {user_id}] Starting stream...")
    start_time = time.time()
    chunks = []

    try:
        async with (
            httpx.AsyncClient(timeout=60.0) as client,
            client.stream(
                "POST", f"{BASE_URL}/api/v1/stream_query", json=payload, headers=headers
            ) as response,
        ):
            if response.status_code != 200:
                print(f"[User {user_id}] Error: {response.status_code}")
                return

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        json_data = httpx.json.loads(data)
                        if "content" in json_data:
                            content = json_data["content"]
                            chunks.append(content)
                            # print(f"[User {user_id}] Chunk: {content}")
                    except Exception:
                        pass
    except Exception as e:
        print(f"[User {user_id}] Exception: {e}")

    duration = time.time() - start_time
    full_text = "".join(chunks)
    print(
        f"[User {user_id}] Finished in {duration:.2f}s. Total chunks: {len(chunks)}. Text length: {len(full_text)}"
    )
    return full_text


async def main():
    # 이 테스트를 실행하기 전에 서버가 실행 중이어야 하며,
    # 테스트용 문서가 'default' 또는 지정된 세션에 업로드되어 있어야 합니다.
    # 여기서는 업로드 단계는 생략하고 동시성 문제 발생 여부만 확인하는 구조입니다.

    print("=== Concurrency Stress Test ===")
    tasks = [
        simulate_streaming_query(1, "Deeply explain the main concept.", "session_1"),
        simulate_streaming_query(2, "Summarize the introduction briefly.", "session_2"),
        simulate_streaming_query(3, "What are the limitations mentioned?", "session_3"),
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
