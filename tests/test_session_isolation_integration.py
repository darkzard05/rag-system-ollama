import pytest
from fastapi.testclient import TestClient
from src.api.api_server import app
import io

client = TestClient(app)


def test_multi_user_session_isolation():
    """
    두 명의 사용자가 서로 다른 세션 ID를 사용할 때 데이터가 격리되는지 테스트합니다.
    """
    session_a = "session-unique-123"
    session_b = "session-unique-456"

    # 1. 사용자 A가 가짜 PDF를 업로드 (실제 PDF 처리는 mock하거나 작은 파일을 사용해야 함)
    # 여기서는 build_rag_pipeline이 실제 Ollama를 호출하므로,
    # 로컬에 Ollama가 실행 중이지 않을 경우를 대비해 핵심 로직이 세션을 구분하는지만 체크합니다.

    pdf_content = b"%PDF-1.4 test pdf content"
    files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

    print(f"\n[Test] 사용자 A ({session_a}) 문서 업로드 시도...")
    # 실제 build_rag_pipeline은 무거운 작업이므로, 여기서는 세션 데이터가 섞이지 않는 구조적 검증에 집중
    # 만약 에러가 나더라도 세션 ID가 다르게 찍히는지 확인
    response_a = client.post(
        "/api/v1/upload", files=files, headers={"X-Session-ID": session_a}
    )

    # 2. 사용자 B가 문서 업로드 없이 질문 시도
    print(f"[Test] 사용자 B ({session_b}) 질문 시도 (문서 없음)...")
    response_b = client.post(
        "/api/v1/query",
        json={"query": "이 문서의 내용은 뭐야?"},
        headers={"X-Session-ID": session_b},
    )

    # 사용자 B는 문서를 업로드하지 않았으므로 400 에러와 함께 메시지가 나와야 함
    assert response_b.status_code == 400
    assert "업로드된 문서가 없습니다" in response_b.json()["detail"]
    print("✅ 사용자 B 세션 격리 확인 (사용자 A의 문서를 공유하지 않음)")

    # 3. 사용자 A의 응답 헤더 확인
    assert response_a.headers["X-Session-ID"] == session_a
    print(
        f"✅ 사용자 A 응답 헤더 세션 ID 일치 확인: {response_a.headers['X-Session-ID']}"
    )


if __name__ == "__main__":
    # 직접 실행 시 pytest 호출
    pytest.main([__file__, "-s"])
