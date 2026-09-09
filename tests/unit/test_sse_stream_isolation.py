"""
스트리밍 오류 격리 단위 테스트 (Phase 2b A5).

한 스트리밍 요청이 중간에 실패해도(SSE [error] + [end] 전송) 공유 세션 상태나
이후 요청을 오염시키지 않아야 한다. 실패 요청 직후의 정상 요청이 완료된다.
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
    api_key = "sk_a5_stream_isolation_test_token"
    api_server.auth_manager.register_fixed_api_key(TEST_USER, api_key)
    return {"Authorization": f"Bearer {api_key}"}


def _ready_session(sid: str) -> None:
    SessionManager.init_session(session_id=sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", session_id=sid)
    SessionManager.set("rag_engine", object(), session_id=sid)


@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_stream_error_does_not_affect_subsequent_request(
    async_client, auth_headers
):
    """실패한 스트리밍 후 동일 세션의 정상 스트리밍이 완료([end])된다."""

    sid = "stream-isolation-test"
    _ready_session(sid)

    async def _fail(self, query, model_name=None):
        async def _gen():
            yield ("custom", {"status": "started"})
            raise KeyError("isolation-boom")

        return _gen()

    with patch.object(api_server.RAGSystem, "astream", _fail):
        async with async_client.stream(
            "POST",
            "/api/v1/stream_query",
            json={"query": "hello", "session_id": sid},
            headers=auth_headers,
        ) as resp:
            assert resp.status_code == 200
            first_text = (await resp.aread()).decode("utf-8")
    assert "event: error" in first_text
    assert "isolation-boom" in first_text

    async def _ok(self, query, model_name=None):
        async def _gen():
            yield ("custom", {"status": "ok"})

        return _gen()

    with patch.object(api_server.RAGSystem, "astream", _ok):
        async with async_client.stream(
            "POST",
            "/api/v1/stream_query",
            json={"query": "hello", "session_id": sid},
            headers=auth_headers,
        ) as resp:
            assert resp.status_code == 200
            second_text = (await resp.aread()).decode("utf-8")

    assert "event: error" not in second_text
    assert "event: end" in second_text
