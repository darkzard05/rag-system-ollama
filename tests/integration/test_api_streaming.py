import pytest
from httpx import AsyncClient, ASGITransport
from src.api.api_server import app, verify_token


# Mock dependency
async def override_verify_token():
    return "test-user"


app.dependency_overrides[verify_token] = override_verify_token


from src.core.session import SessionManager
from unittest.mock import MagicMock


@pytest.fixture
def initialized_session():
    sid = "test-session"
    SessionManager.init_session(session_id=sid)
    SessionManager.set("last_uploaded_file_name", "test.pdf", session_id=sid)
    SessionManager.set("rag_engine", MagicMock(), session_id=sid)
    return sid


@pytest.mark.asyncio
async def test_streaming_endpoint_returns_sse(initialized_session):
    """Test that /api/v1/stream_query returns valid SSE stream."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Note: This test requires a running Ollama service or proper mocking
        # Assuming the system has appropriate mocks or test-mode configuration
        response = await client.post(
            "/api/v1/stream_query",
            json={"query": "cm3가 뭔가요?", "session_id": initialized_session},
        )
        # We expect 200, but if it hits the actual RAG code it will fail with an error because mocks aren't fully set up.
        # However, checking if endpoint is reachable (200) is the first step.
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Verify SSE format
        content = response.text
        assert "data:" in content or "event:" in content
