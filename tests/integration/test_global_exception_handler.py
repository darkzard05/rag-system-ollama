from fastapi.testclient import TestClient
from src.api.api_server import app

client = TestClient(app)


def test_global_exception_handler():
    """
    정의된 커스텀 예외가 전역 핸들러에 의해 구조화된 JSON으로 변환되는지 테스트합니다.
    """
    # 1. 문서 없이 질문 (SessionError 유발)
    response = client.post(
        "/api/v1/query",
        json={"query": "테스트 질문"},
        headers={"X-Session-ID": "test-error-session"},
    )

    data = response.json()
    assert response.status_code == 400
    assert data["error_code"] == "SESSION_ERROR"
    assert "업로드된 문서가 없습니다" in data["message"]

    # 2. 잘못된 확장자 업로드 (EmptyPDFError 유발)
    import io

    files = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
    response = client.post(
        "/api/v1/upload", files=files, headers={"X-Session-ID": "test-error-session"}
    )

    data = response.json()
    assert response.status_code == 400
    assert data["error_code"] == "EMPTY_PDF"
