"""
API 인증 결함 검증 테스트 (TDD - RED)

검증 대상 결함:
A. POST /api/v1/login 엔드포인트 부재 (404/405)
B. 로그인으로 발급된 JWT가 보호 엔드포인트 인증에 사용 가능한가
C. 로그아웃/폐기 후 동일 토큰이 401로 거부되는가 (-store 미확인 문제)
D. 세션 소유권 미검사 - 다른 사용자가 타 사용자 세션에 접근 가능
E. /api/v1/admin/stats 가 관리자 전용이 아님 (role 미검사)

테스트 환경: Ollama 불필요. 업로드 시 RAGSystem/get_embedder 를 모킹합니다.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.api.api_server import TEST_PASSWORD as ADMIN_PASSWORD
from src.api.api_server import app, verify_token

from core.session import SessionManager

client = TestClient(app)

ADMIN_USERNAME = "admin_user"


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """세션 상태를 테스트 간 격리합니다."""
    SessionManager.reset()
    yield
    SessionManager.reset()


@pytest.fixture(autouse=True)
def clean_verify_token_override():
    """다른 테스트 모듈이 남긴 verify_token 오버라이드가 인증 검증을 우회하지 않도록 제거합니다."""
    app.dependency_overrides.pop(verify_token, None)
    yield
    app.dependency_overrides.pop(verify_token, None)


def _login(username: str, password: str):
    """(GREEN 후) /api/v1/login 을 통해 토큰을 획득합니다."""
    resp = client.post(
        "/api/v1/login", json={"username": username, "password": password}
    )
    return resp


def _issue_token(user_id: str, username: str, password: str) -> str:
    """로그인 엔드포인트가 발급하는 것과 동일한 토큰을 auth_manager로 직접 발급합니다.

    (RED 단계에서 /login 이 없더라도 소유권/역할 결함을 격리 검증하기 위함)
    """
    from src.api.api_server import auth_manager

    if user_id not in auth_manager._users:  # type: ignore[attr-defined]
        assert auth_manager.register_user(user_id, username, password)
    result = auth_manager.authenticate(user_id, password)
    assert result is not None, f"인증 실패: {user_id}"
    return result[0]


# --- A. 로그인 엔드포인트 존재 및 토큰 발급 ---


def test_login_returns_token_for_admin():
    """관리자 자격증명으로 로그인 시 200 + access_token 을 반환해야 한다."""
    resp = _login(ADMIN_USERNAME, ADMIN_PASSWORD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"]
    assert data["token_type"] == "bearer"


def test_login_rejects_wrong_password():
    """잘못된 비밀번호는 401 을 반환해야 한다."""
    resp = _login(ADMIN_USERNAME, "wrong-password")
    assert resp.status_code == 401


# --- B. 로그인 토큰이 보호 엔드포인트에서 동작 ---


def test_login_token_authorizes_protected_endpoint():
    """로그인 토큰을 Bearer 헤더로 보호 엔드포인트(GET /api/v1/admin/stats)에 사용할 수 있어야 한다."""
    resp = _login(ADMIN_USERNAME, ADMIN_PASSWORD)
    assert resp.status_code == 200
    token = resp.json()["access_token"]

    stats_resp = client.get(
        "/api/v1/admin/stats", headers={"Authorization": f"Bearer {token}"}
    )
    assert stats_resp.status_code == 200


# --- C. 로그아웃/폐기 후 토큰 무효화 ---


def test_logout_revokes_token():
    """로그아웃 후 동일 토큰은 401 로 거부되어야 한다 (토큰 스토어 확인)."""
    resp = _login(ADMIN_USERNAME, ADMIN_PASSWORD)
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    before = client.get("/api/v1/admin/stats", headers=headers)
    assert before.status_code == 200

    logout_resp = client.post("/api/v1/logout", headers=headers)
    assert logout_resp.status_code == 200

    after = client.get("/api/v1/admin/stats", headers=headers)
    assert after.status_code == 401


# --- D. 세션 소유권 검사 ---


def test_other_user_cannot_delete_or_query_owned_session():
    """B 세션 소유자가 A 세션에 접근/삭제하면 거부(403)되어야 한다."""
    jwt_admin = _issue_token("admin", ADMIN_USERNAME, ADMIN_PASSWORD)

    other_uid = "regular_b"
    other_username = "regular_user"
    other_password = "user-b-pass"
    jwt_other = _issue_token(other_uid, other_username, other_password)

    session_id = "admin-owned-session"

    # 관리자가 세션을 생성(업로드)하여 소유권을 얻는다.
    mock_rag_system = MagicMock()
    mock_rag_system.return_value.build_pipeline = AsyncMock(
        return_value=("인덱싱 완료", False)
    )
    pdf = {"file": ("test.pdf", io.BytesIO(b"%PDF-1.4 mock"), "application/pdf")}
    with (
        patch("src.api.api_server.RAGSystem", mock_rag_system),
        patch(
            "src.api.api_server.RAGResourceManager.get_embedder",
            new_callable=AsyncMock,
        ),
    ):
        upload = client.post(
            "/api/v1/upload",
            files=pdf,
            data={"session_id": session_id},
            headers={"Authorization": f"Bearer {jwt_admin}"},
        )
    assert upload.status_code == 200, upload.text

    other_headers = {"Authorization": f"Bearer {jwt_other}"}

    # 다른 사용자가 소유 세션을 삭제하려 하면 403
    del_resp = client.delete(f"/api/v1/session/{session_id}", headers=other_headers)
    assert del_resp.status_code == 403

    # 다른 사용자가 소유 세션에 질의하면 403
    query_resp = client.post(
        "/api/v1/query",
        json={"query": "도청 질문", "session_id": session_id},
        headers=other_headers,
    )
    assert query_resp.status_code == 403

    # 소유자(관리자) 본인은 삭제 가능 (positive control)
    admin_del = client.delete(
        f"/api/v1/session/{session_id}",
        headers={"Authorization": f"Bearer {jwt_admin}"},
    )
    assert admin_del.status_code == 200


# --- E. /api/v1/admin/stats 는 관리자 전용 ---


def test_stats_requires_admin_role():
    """비관리자 인증 사용자는 /api/v1/admin/stats 에 403 을 받아야 한다."""
    non_admin_uid = "non_admin_x"
    non_admin_uname = "regular_user"
    non_admin_pass = "non-admin-pass"
    jwt_non_admin = _issue_token(non_admin_uid, non_admin_uname, non_admin_pass)

    resp = client.get(
        "/api/v1/admin/stats",
        headers={"Authorization": f"Bearer {jwt_non_admin}"},
    )
    assert resp.status_code == 403

    # 관리자는 접근 가능 (positive control)
    jwt_admin = _issue_token("admin", ADMIN_USERNAME, ADMIN_PASSWORD)
    admin_resp = client.get(
        "/api/v1/admin/stats",
        headers={"Authorization": f"Bearer {jwt_admin}"},
    )
    assert admin_resp.status_code == 200
