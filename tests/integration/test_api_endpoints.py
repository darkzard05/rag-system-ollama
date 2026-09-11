"""
Modern Async API Tests using HTTPX and Pytest-Asyncio.
Tests FastAPI endpoints directly without running a separate uvicorn server.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from src.api import api_server
from src.api.api_server import app

# --- Fixtures ---


@pytest.fixture
def auth_headers():
    """인증 헤더 생성"""
    # api_server.py에서 초기화된 auth_manager에 직접 키를 등록하여 확실하게 인증 통과
    api_key = "sk_admin_test_token_12345"
    from src.api.api_server import TEST_USER, auth_manager

    auth_manager.register_fixed_api_key(TEST_USER, api_key)
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def mock_rag_resources():
    """RAG 리소스(LLM, Embedder 등)를 모킹하여 무거운 로딩 방지"""
    with (
        patch(
            "src.core.resource_manager.ResourceManager.get_llm_for_session",
            new_callable=AsyncMock,
        ) as mock_get_llm,
        patch(
            "src.core.resource_manager.ResourceManager.get_embedder_for_session",
            new_callable=AsyncMock,
        ) as mock_get_embedder,
    ):
        # Mock LLM
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.model_name = "mock-embedding-model"
        mock_get_embedder.return_value = mock_embedder

        yield mock_get_llm


@pytest.fixture
def mock_session_manager():
    """SessionManager 상태 모킹"""
    # api_server 모듈 객체의 속성을 직접 패치하여 확실하게 적용
    with (
        patch.object(api_server, "SessionManager") as mock_sm,
        patch.object(
            api_server.RAGSystem, "astream", new_callable=AsyncMock
        ) as mock_astream,
    ):
        # Mock QA Chain (LangChain Runnable)
        mock_chain = MagicMock()

        async def async_gen(*args, **kwargs):
            events = [
                {
                    "event": "on_custom_event",
                    "name": "response_chunk",
                    "data": {"chunk": "Hello"},
                },
                {
                    "event": "on_custom_event",
                    "name": "response_chunk",
                    "data": {"chunk": " "},
                },
                {
                    "event": "on_custom_event",
                    "name": "response_chunk",
                    "data": {"chunk": "World"},
                },
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

        async def astream_impl(query: str, model_name: str | None = None):
            async def _stream():
                async for event in async_gen():
                    if event["event"] == "on_custom_event":
                        yield ("custom", {"content": event["data"].get("chunk", "")})
                    elif event["event"] == "on_chain_end":
                        yield (
                            "custom",
                            {
                                "documents": event["data"]
                                .get("output", {})
                                .get("documents", [])
                            },
                        )

            return _stream()

        mock_astream.side_effect = astream_impl

        # get 메서드의 동작 정의
        def get_side_effect(key, default=None, **kwargs):
            if key == "pdf_processed":
                return True
            if key == "rag_engine":  # api_server.py에서는 rag_engine을 사용함
                return mock_chain
            if key == "last_uploaded_file_name":
                return "test.pdf"
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


@pytest.mark.asyncio
async def test_stream_query_success(
    async_client, mock_rag_resources, mock_session_manager, auth_headers
):
    """스트리밍 질의 성공 시나리오 테스트"""
    payload = {"query": "Hello?", "use_cache": True}

    async with async_client.stream(
        "POST", "/api/v1/stream_query", json=payload, headers=auth_headers
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
                        # JSON 데이터인 경우 (source 등) 파싱하여 텍스트 추출
                        data = json.loads(content)
                        if "content" in data:
                            received_text += data["content"]
                    except json.JSONDecodeError:
                        received_text += content

        assert "Hello World" in received_text


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="build_rag_pipeline → RAGSystem.build_pipeline으로 리팩토링됨"
)
async def test_upload_flow_mocked(async_client, mock_rag_resources, auth_headers):
    """파일 업로드 엔드포인트 테스트 (Mocked)"""

    # build_rag_pipeline 함수도 모킹해야 함 (Core 로직 실행 방지)
    with patch("src.api.api_server.build_rag_pipeline") as mock_build:
        mock_build.return_value = ("Success", False)

        files = {"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
        response = await async_client.post(
            "/api/v1/upload", files=files, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert "message" in data


def _parse_sse_frames(lines: list[str]) -> list[tuple[str | None, Any]]:
    """SSE line 목록을 (event_type, data) 프레임 목록으로 파싱합니다."""
    frames: list[tuple[str | None, Any]] = []
    current_event: str | None = None
    for line in lines:
        if line.startswith("event: "):
            current_event = line[len("event: ") :].strip()
        elif line.startswith("data: "):
            raw = line[len("data: ") :]
            try:
                frames.append((current_event, json.loads(raw)))
            except json.JSONDecodeError:
                frames.append((current_event, raw))
    return frames


def _assert_end_is_last_frame(
    frames: list[tuple[str | None, Any]],
) -> None:
    """마지막 프레임이 ``event: end`` (status=done) 임을 검증합니다."""
    last_event, last_data = frames[-1]
    assert last_event == "end"
    assert last_data == {"status": "done"}


@pytest.mark.asyncio
async def test_stream_query_emits_citations_and_metrics(
    async_client, mock_rag_resources, mock_session_manager, auth_headers
):
    """``citations`` / ``metrics`` 이벤트가 SSE 프레임으로 전송되는지 검증"""
    events = [
        ("custom", {"citations": [{"source": "S", "page": 1}]}),
        (
            "updates",
            {"generate": {"performance": {"token_count": 5, "input_token_count": 3}}},
        ),
    ]

    async def astream_impl(query: str, model_name: str | None = None):
        async def _stream():
            for mode, data in events:
                yield (mode, data)
                await asyncio.sleep(0.005)

        return _stream()

    with patch.object(
        api_server.RAGSystem, "astream", new_callable=AsyncMock
    ) as mock_astream:
        mock_astream.side_effect = astream_impl

        payload = {"query": "인용과 성능을 요청합니다", "use_cache": True}
        async with async_client.stream(
            "POST", "/api/v1/stream_query", json=payload, headers=auth_headers
        ) as response:
            assert response.status_code == 200
            lines = [line async for line in response.aiter_lines()]

    frames = _parse_sse_frames(lines)
    event_types = [event for event, _ in frames]

    assert "citations" in event_types
    citations_frames = [data for event, data in frames if event == "citations"]
    assert citations_frames, "event: citations 프레임이 존재해야 합니다."
    assert citations_frames[0] == {"citations": [{"source": "S", "page": 1}]}

    assert "metrics" in event_types
    metrics_frames = [data for event, data in frames if event == "metrics"]
    assert metrics_frames, "event: metrics 프레임이 존재해야 합니다."
    matched = any(
        data.get("metrics", {}).get("token_count") == 5
        and data.get("metrics", {}).get("input_token_count") == 3
        for data in metrics_frames
    )
    assert matched, "metrics 페이로드에 token_count/input_token_count가 있어야 합니다."

    assert "error" not in event_types
    _assert_end_is_last_frame(frames)


@pytest.mark.asyncio
async def test_stream_query_emits_keepalive_on_stall(
    async_client, mock_rag_resources, mock_session_manager, auth_headers
):
    """스트림이 지연(stall)될 때 keepalive comment 프레임을 발행하는지 검증"""
    with patch("api.routes_chat.SSE_KEEPALIVE_INTERVAL_SECONDS", 0.05):

        async def stall_astream_impl(query: str, model_name: str | None = None):
            async def _stream():
                await asyncio.sleep(0.2)
                yield ("custom", {"content": "hello"})

            return _stream()

        with patch.object(
            api_server.RAGSystem, "astream", new_callable=AsyncMock
        ) as mock_astream:
            mock_astream.side_effect = stall_astream_impl

            payload = {"query": "지연 응답 요청", "use_cache": True}
            async with async_client.stream(
                "POST", "/api/v1/stream_query", json=payload, headers=auth_headers
            ) as response:
                assert response.status_code == 200
                lines = [line async for line in response.aiter_lines()]

    keepalives = [i for i, line in enumerate(lines) if line == ": keep-alive"]
    assert keepalives, "stall 동안 keepalive 프레임이 발행되어야 합니다."
    first_data_idx = next(
        i for i, line in enumerate(lines) if line.startswith("data: ")
    )
    assert all(i < first_data_idx for i in keepalives), (
        "keepalive는 첫 데이터 프레임보다 앞서야 합니다."
    )

    frames = _parse_sse_frames(lines)
    event_types = [event for event, _ in frames]
    assert "error" not in event_types
    _assert_end_is_last_frame(frames)


@pytest.mark.asyncio
async def test_stream_query_no_keepalive_when_flowing(
    async_client, mock_rag_resources, mock_session_manager, auth_headers
):
    """스트림이 원활히 흐르면 keepalive comment 프레임이 없어야 한다"""
    with patch("api.routes_chat.SSE_KEEPALIVE_INTERVAL_SECONDS", 0.05):

        async def flowing_astream_impl(query: str, model_name: str | None = None):
            async def _stream():
                for content in ("alpha ", "beta ", "gamma"):
                    yield ("custom", {"content": content})
                    await asyncio.sleep(0.005)

            return _stream()

        with patch.object(
            api_server.RAGSystem, "astream", new_callable=AsyncMock
        ) as mock_astream:
            mock_astream.side_effect = flowing_astream_impl

            payload = {"query": "연속 응답 요청", "use_cache": True}
            async with async_client.stream(
                "POST", "/api/v1/stream_query", json=payload, headers=auth_headers
            ) as response:
                assert response.status_code == 200
                lines = [line async for line in response.aiter_lines()]

    assert not any(line == ": keep-alive" for line in lines), (
        "흐르는 스트림에는 keepalive 프레임이 없어야 합니다."
    )
    assert any(line.startswith("data: ") for line in lines), (
        "콘텐츠 데이터 프레임이 존재해야 합니다."
    )
    frames = _parse_sse_frames(lines)
    event_types = [event for event, _ in frames]
    assert "error" not in event_types
    _assert_end_is_last_frame(frames)
