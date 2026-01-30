import pytest
from fastapi.testclient import TestClient
from src.api.api_server import app

client = TestClient(app)


def test_global_exception_handler():
    """
    정의된 커스텀 예외가 전역 핸들러에 의해 구조화된 JSON으로 변환되는지 테스트합니다.
    """
    # 1. 문서 없이 질문 (SessionError 유발)
    print("\n[Test] 문서 없이 질문 시도 (SessionError 유발)...\n")
    response = client.post(
        "/api/v1/query",
        json={"query": "테스트 질문"},
        headers={"X-Session-ID": "test-error-session"},
    )

    data = response.json()
    assert response.status_code == 400
    assert data["error_code"] == "SESSION_ERROR"
    assert "업로드된 문서가 없습니다" in data["message"]
    print("✅ SessionError 핸들링 확인")

    # 2. 잘못된 확장자 업로드 (EmptyPDFError 유발)
    print("[Test] 잘못된 파일 확장자 업로드 시도 (EmptyPDFError 유발)...\n")
    import io

    files = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
    response = client.post(
        "/api/v1/upload", files=files, headers={"X-Session-ID": "test-error-session"}
    )

    data = response.json()
    assert response.status_code == 400
    assert data["error_code"] == "EMPTY_PDF"
    print("✅ EmptyPDFError 핸들링 확인")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
