from fastapi.testclient import TestClient
from src.api.api_server import TEST_USER, app, auth_manager

client = TestClient(app)

API_KEY = "sk_admin_test_exception_handler"


def test_global_exception_handler():
    """
    인증된 요청에 대해 문서가 없는 잘못된 요청이 400 으로 거부되는지 테스트합니다.

    (수정: 이전에는 인증 없이 요청하여 401 을 받았습니다. 전역 예외 처리 동작을
    검증하기 위해 먼저 유효한 토큰/API 키를 획득합니다.)
    """
    auth_manager.register_fixed_api_key(TEST_USER, API_KEY)
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # 1. 문서 없이 질문 (SessionError 유발)
    response = client.post(
        "/api/v1/query",
        json={"query": "테스트 질문"},
        headers={**headers, "X-Session-ID": "test-error-session"},
    )

    data = response.json()
    assert response.status_code == 400
    assert "먼저 문서를 업로드" in data["detail"]

    # 2. 잘못된 확장자 업로드 (EmptyPDFError 유발)
    import io

    files = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
    response = client.post(
        "/api/v1/upload",
        files=files,
        headers={**headers, "X-Session-ID": "test-error-session"},
    )

    data = response.json()
    assert response.status_code == 400
    assert "PDF 파일만 업로드" in data["detail"]
