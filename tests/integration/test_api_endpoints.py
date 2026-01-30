"""
Modern Async API Tests using HTTPX and Pytest-Asyncio.
Tests FastAPI endpoints directly without running a separate uvicorn server.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from src.api import api_server
from src.api.api_server import app

# --- Fixtures ---


@pytest.fixture
def mock_rag_resources():
    """RAG 리소스(LLM, Embedder 등)를 모킹하여 무거운 로딩 방지"""
    with patch("src.api.api_server.RAGResourceManager") as mock_mgr:
        # Mock LLM
        mock_llm = MagicMock()
        mock_mgr.get_llm = AsyncMock(return_value=mock_llm)

        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.model_name = "mock-embedding-model"
        mock_mgr.get_embedder = AsyncMock(return_value=mock_embedder)

        yield mock_mgr


@pytest.fixture
def mock_session_manager():
    """SessionManager 상태 모킹"""
    # api_server 모듈 객체의 속성을 직접 패치하여 확실하게 적용
    with patch.object(api_server, "SessionManager") as mock_sm:
        # Mock QA Chain (LangChain Runnable)
        mock_chain = MagicMock()

        async def async_gen(*args, **kwargs):
            events = [
                {"event": "on_parser_stream", "data": {"chunk": "Hello"}},
                {"event": "on_parser_stream", "data": {"chunk": " "}},
                {"event": "on_parser_stream", "data": {"chunk": "World"}},
                {
                    "event": "on_chain_end",
                    "name": "retrieve",
                    "data": {"output": {"documents": []}},
                },
            ]
            for event in events:
                yield event
                await asyncio.sleep(0.01)

        mock_chain.astream_events = async_gen

        # get 메서드의 동작 정의
        def get_side_effect(key, default=None):
            if key == "pdf_processed":
                return True
            if key == "qa_chain":
                return mock_chain
            return default

        mock_sm.get.side_effect = get_side_effect

        yield mock_sm


@pytest_asyncio.fixture
async def async_client():
    """FastAPI 앱에 연결된 비동기 클라이언트 생성"""
    # ASGITransport를 사용하여 앱 직접 연결
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- Tests ---


@pytest.mark.asyncio
async def test_health_check(async_client):
    """서버 헬스 체크 테스트"""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "session_id" in data


@pytest.mark.asyncio
async def test_stream_query_success(
    async_client, mock_rag_resources, mock_session_manager
):
    """스트리밍 질의 성공 시나리오 테스트"""
    payload = {"query": "Hello?", "use_cache": True}

    async with async_client.stream(
        "POST", "/api/v1/stream_query", json=payload
    ) as response:
        assert response.status_code == 200
        # charset=utf-8이 붙을 수 있으므로 startswith 사용
        assert response.headers["content-type"].startswith("text/event-stream")

        received_text = ""
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                content = line[6:]
                if content != "[DONE]":
                    try:
                        # JSON 데이터인 경우 (source 등) 제외하고 텍스트만
                        json.loads(content)
                    except json.JSONDecodeError:
                        received_text += content

        assert "Hello World" in received_text


@pytest.mark.asyncio
async def test_upload_flow_mocked(async_client, mock_rag_resources):
    """파일 업로드 엔드포인트 테스트 (Mocked)"""

    # build_rag_pipeline 함수도 모킹해야 함 (Core 로직 실행 방지)
    with patch("src.api.api_server.build_rag_pipeline") as mock_build:
        mock_build.return_value = ("Success", False)

        files = {"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
        response = await async_client.post("/api/v1/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert "message" in data
