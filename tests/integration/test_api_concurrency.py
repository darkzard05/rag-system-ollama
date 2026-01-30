import asyncio
import os
import time

import httpx

# API 서버 주소
BASE_URL = "http://127.0.0.1:8000"


async def test_user_session(user_id: int, pdf_path: str, query: str):
    """개별 사용자 세션 시뮬레이션"""
    session_id = f"user_{user_id}_{int(time.time())}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. 문서 업로드
        print(f"[User {user_id}] Uploading document...")
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            data = {"session_id": session_id}
            resp = await client.post(
                f"{BASE_URL}/api/v1/upload", files=files, data=data
            )
            assert resp.status_code == 200, f"Upload failed for user {user_id}"

        # 2. 질의 수행
        print(f"[User {user_id}] Querying: {query}")
        query_data = {"query": query, "session_id": session_id, "use_cache": True}
        resp = await client.post(f"{BASE_URL}/api/v1/query", json=query_data)
        if resp.status_code == 422:
            print(f"[User {user_id}] 422 Error Detail: {resp.text}")
        assert resp.status_code == 200, (
            f"Query failed for user {user_id} with status {resp.status_code}"
        )

        result = resp.json()
        answer = result["answer"]
        print(f"[User {user_id}] Received answer (len: {len(answer)})")

        # 3. 무결성 검증 (자신의 문서 내용이 포함되어 있는지 간단히 확인)
        # 이 부분은 실제 PDF 내용에 따라 조정이 필요하지만, 여기서는 에러 없이 응답이 왔는지만 확인
        return session_id, answer


async def main():
    # 테스트용 PDF 경로 (기존에 존재하는 파일 활용)
    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        print("Test PDF not found. Please ensure tests/data/2201.07520v1.pdf exists.")
        return

    print("=== API 동시성 및 세션 격리 테스트 시작 ===")
    start_time = time.time()

    # 3명의 사용자가 동시에 서로 다른 세션으로 접근 시뮬레이션
    tasks = [
        test_user_session(1, pdf_path, "What is the primary topic of this paper?"),
        test_user_session(
            2, pdf_path, "Explain the methodology used in this research."
        ),
        test_user_session(3, pdf_path, "Summarize the key findings."),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"User {i + 1} failed with error: {res}")
        else:
            sid, ans = res
            print(f"User {i + 1} (Session: {sid}) success.")

    duration = time.time() - start_time
    print(f"\n=== 테스트 완료 (소요 시간: {duration:.2f}s) ===")


if __name__ == "__main__":
    # 서버가 실행 중이어야 합니다 (uvicorn src.api.api_server:app)
    try:
        asyncio.run(main())
    except ConnectionError:
        print(
            "API server is not running. Please start it with 'python src/api/api_server.py'"
        )
