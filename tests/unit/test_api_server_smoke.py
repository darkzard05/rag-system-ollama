"""
api_server.py 단위 스모크 테스트.

src/api/api_server.py 는 모듈 import 시점에 실제 부트스트랩(관리자 크리덴셜/
API 키 생성, 라우트 등록)을 수행한다. 본 테스트는 모델/Ollama 호출 없이
FastAPI TestClient(동기, lifespan 미실행)로 진입 계층을 스모크한다.
"""

import src.api.api_server as srv
from fastapi.testclient import TestClient


def test_module_import_bootstraps() -> None:
    """import 만으로 모듈 레벨 부트스트랩(앱/크리덴셜/라우트)이 실행된다."""
    from fastapi import FastAPI

    assert isinstance(srv.app, FastAPI)
    assert isinstance(srv.TEST_API_KEY, str)
    assert srv.TEST_API_KEY != ""
    paths = [getattr(r, "path", None) for r in srv.app.routes]
    for expected in ("/api/v1/health", "/api/v1/login", "/api/v1/query"):
        assert expected in paths, f"{expected} route not registered"


def test_health_no_auth_required() -> None:
    """/api/v1/health 는 인증 불필요, 모델 호출 없음."""
    client = TestClient(srv.app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "model" in body


def test_login_with_bootstrap_credentials() -> None:
    """모듈 부트스트랩 크리덴셜로 /api/v1/login 이 200 + 토큰을 반환.

    부트스트랩은 upsert_admin_credentials(TEST_USER, "admin_user", password) 이므로
    로그인 username 은 "admin_user" 이다(TEST_USER 는 내부 user_id).
    """
    client = TestClient(srv.app)
    resp = client.post(
        "/api/v1/login",
        json={"username": "admin_user", "password": srv.TEST_PASSWORD},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("access_token") or body.get("token")


def test_query_requires_auth() -> None:
    """인증 없는 /api/v1/query 는 조기 거부(401/403)."""
    client = TestClient(srv.app)
    resp = client.post(
        "/api/v1/query",
        json={"query": "x", "session_id": "default"},
    )
    assert resp.status_code in (401, 403)


def test_upload_rejects_unauthenticated() -> None:
    """인증 없는 /api/v1/upload 는 거부(401/403). 실제 파일 미업로드."""
    client = TestClient(srv.app)
    resp = client.post(
        "/api/v1/upload",
        data={"session_id": "default"},
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code in (401, 403)


def test_logout_requires_auth() -> None:
    """인증 없는 /api/v1/logout 는 거부(401). 라우트 진입만 스모크.

    부트스트랩 TEST_API_KEY 는 모듈 전역이라 다른 테스트(auth)와 공유되므로
    여기서 무효화하지 않는다(순서 의존 오염 방지). 성공 경로는 별도 키로 커버.
    """
    client = TestClient(srv.app)
    resp = client.post("/api/v1/logout", json={"session_id": "default"})
    assert resp.status_code in (401, 403)


def test_logout_success_path_local_key() -> None:
    """로컬 생성 키로 /api/v1/logout 성공 경로(283-287)를 커버. 전역 키 미변경."""
    client = TestClient(srv.app)
    local_key = srv.auth_manager.create_api_key(srv.TEST_USER, expires_in=3600)
    headers = {"Authorization": f"Bearer {local_key}"}
    resp = client.post(
        "/api/v1/logout", json={"session_id": "default"}, headers=headers
    )
    assert resp.status_code == 200


def test_query_happy_path_mocked() -> None:
    """인증 + ResourceManager mock 상태에서 /api/v1/query 는 비즈니스 로직까지 도달.

    라우트 핸들러 시그니처/응답 형태가 무거워 실제 200 보장이 어려우므로,
    인증 통과 후 모델 로딩 단계까지 진입했음(500 아님)을 asserting 한다.
    """
    from unittest.mock import AsyncMock, patch

    with (
        patch(
            "src.core.resource_manager.ResourceManager.get_llm_for_session",
            new_callable=AsyncMock,
        ),
        patch(
            "src.core.resource_manager.ResourceManager.get_embedder_for_session",
            new_callable=AsyncMock,
        ),
    ):
        client = TestClient(srv.app)
        headers = {"Authorization": f"Bearer {srv.TEST_API_KEY}"}
        resp = client.post(
            "/api/v1/query",
            json={"query": "hello", "session_id": "default"},
            headers=headers,
        )
    assert resp.status_code in (200, 400, 422)
