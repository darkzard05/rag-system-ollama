"""
SSE 스트리밍/비스트리밍 오류 경로 단위 테스트 (Phase 2b A5).

- 스트리밍: 예상치 못한 예외가 발생해도 [error] → [end] SSE 이벤트로
  fail-closed 되며 응답이 중간에 끊기지 않는다.
- 비스트리밍(query): 예외는 500 JSON 응답으로, SSE 프레이밍이 없다.
"""

from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from src.api import api_server
from src.api.api_server import TEST_USER, app

from core.session import SessionManager


@pytest.fixture
def auth_headers():
    """인증 헤더: 고정 API 키를 등록해 인증 통과."""
    api_key = "sk_a5_sse_fallback_test_token"
    api_server.auth_manager.register_fixed_api_key(TEST_USER, api_key)
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def session_state():
    """세션 state 를 준비해 질의 라우트가 파일 업로드/엔진 검사를 통과하도록 한다."""
    sid = "default"
    SessionManager.init_session(session_id=sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", session_id=sid)
    SessionManager.set("rag_engine", object(), session_id=sid)
    return sid


@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_stream_unexpected_exception_yields_sse_error_not_500(
    async_client, auth_headers, session_state
):
    """스트리밍 중 예상치 못한 예외(기존 tuple 밖, KeyError)가 발생하면
    HTTP 200 시작 + [error] 이벤트 + [end] 이벤트로 덮여진다(500 아님)."""

    async def _raise(self, query, model_name=None):
        async def _gen():
            yield ("custom", {"status": "started"})
            raise KeyError("boom-unexpected")

        return _gen()

    with patch.object(api_server.RAGSystem, "astream", _raise):
        async with async_client.stream(
            "POST",
            "/api/v1/stream_query",
            json={"query": "hello", "session_id": session_state},
            headers=auth_headers,
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = await resp.aread()
            text = body.decode("utf-8")

    assert "event: error" in text
    assert "boom-unexpected" in text
    assert "event: end" in text


@pytest.mark.asyncio
async def test_query_rag_error_returns_500_without_sse_events(
    async_client, auth_headers, session_state
):
    """비스트리밍 query_rag: ValueError 는 500 JSON 응답, SSE 프레이밍 없음."""

    async def _raise(self, query, model_name=None):
        raise ValueError("query boom")

    with patch.object(api_server.RAGSystem, "aquery", _raise):
        resp = await async_client.post(
            "/api/v1/query",
            json={"query": "hello", "session_id": session_state},
            headers=auth_headers,
        )

    assert resp.status_code == 500
    assert not resp.headers.get("content-type", "").startswith("text/event-stream")
    assert "event:" not in resp.text
    assert "data:" not in resp.text
    assert "detail" in resp.json()
