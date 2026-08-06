"""
SSE 스트리밍 오류 격리 검증 테스트 (TDD - RED)

검증 대상 결함:
- /api/v1/stream_query 의 event_generator 는 (RuntimeError, ValueError,
  ConnectionError) 만 캐치합니다. PDFProcessingError 계열(VectorStoreError 등)이
  제너레이터 내부(RAGSystem.astream)에서 발생하면 SSE 스트림이 끊기고
  처리되지 않은 예외(ExceptionGroup)로 전파됩니다.
- 기대 동작: 200 SSE 응답으로 시작한 스트림이 'error' SSE 이벤트로 우아하게 종료되어야 합니다.

테스트 환경: Ollama 불필요. RAGSystem.astream 만 모킹합니다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from src.api import api_server
from src.api.api_server import app

from common.exceptions import VectorStoreError


@pytest.fixture
def auth_headers():
    """인증 헤더 생성"""
    api_key = "sk_admin_test_token_12345"
    from src.api.api_server import TEST_USER, auth_manager

    auth_manager.register_fixed_api_key(TEST_USER, api_key)
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def mock_session_manager():
    """SessionManager 상태 모킹 (rag_engine/last_uploaded_file_name 사전 체크 통과용)"""
    with patch.object(api_server, "SessionManager") as mock_sm:
        mock_chain = MagicMock()

        def get_side_effect(key, default=None, **kwargs):
            if key == "pdf_processed":
                return True
            if key == "rag_engine":
                return mock_chain
            if key == "last_uploaded_file_name":
                return "test.pdf"
            return default

        mock_sm.get.side_effect = get_side_effect

        yield mock_sm


@pytest_asyncio.fixture
async def async_client():
    """FastAPI 앱에 연결된 비동기 클라이언트 생성"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_vector_store_error_yields_sse_error_event(
    async_client, mock_session_manager, auth_headers
):
    """VectorStoreError가 event_generator 내부에서 발생하면 SSE error 이벤트로 처리되어야 한다.

    event_generator 는 `await rag_sys.astream(...)` 결과를 stream_graph_events 로
    소비하므로, RAGSystem.astream 이 VectorStoreError("파일 해시 없음")를
    던지면 현재 코드에서는 (RuntimeError, ValueError, ConnectionError)에
    포함되지 않아 스트림이 끊긴다. 수정 후에는 'error' SSE 이벤트로 격리되어야 한다.
    """
    with patch.object(
        api_server.RAGSystem, "astream", new_callable=AsyncMock
    ) as mock_astream:
        mock_astream.side_effect = VectorStoreError("파일 해시 없음")

        resp = await async_client.post(
            "/api/v1/stream_query",
            json={"query": "hello", "session_id": "err-sess"},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "파일 해시 없음" in resp.text
