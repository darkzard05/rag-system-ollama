import requests


def test_streaming_api():
    base_url = "http://127.0.0.1:8000/api/v1"

    print("1. 서버 상태 확인...")
    try:
        res = requests.get(f"{base_url}/health")
        print(f"상태: {res.json()}")
    except:
        print(
            "에러: API 서버가 실행 중인지 확인하세요 (uvicorn src.api.api_server:app)"
        )
        return

    # 실제 업로드는 생략 (이미 UI나 이전 테스트에서 업로드된 파일이 세션에 있다고 가정)
    # 만약 세션이 비어있다면 에러가 발생할 것이므로, 이를 확인하는 용도로 사용

    print("\n2. 스트리밍 질의 시작...")
    payload = {"query": "이 문서의 주요 내용을 요약해줘.", "use_cache": True}

    try:
        # stream=True 옵션으로 SSE 스트림 수신
        with requests.post(
            f"{base_url}/stream_query", json=payload, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"에러 발생: {response.text}")
                return

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        content = decoded_line[6:]
                        print(content, end="", flush=True)
                    elif decoded_line.startswith("event: "):
                        print(f"\n[{decoded_line[7:]}] ", end="")
    except Exception as e:
        print(f"\n테스트 중 오류: {e}")


if __name__ == "__main__":
    test_streaming_api()
