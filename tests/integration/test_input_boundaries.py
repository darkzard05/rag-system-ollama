"""
API 입력 경계 검증 테스트 (F11 - TDD).

검증 대상:
- 업로드 파일 크기 상한(MAX_PDF_SIZE_BYTES, 50MB) 초과 시 413
- 질문 길이 상한(MAX_QUERY_LENGTH, 4000자) 초과 시 400
- 세션 ID 길이 상한(MAX_SESSION_ID_LENGTH, 64자) 초과 시 400

테스트 환경: Ollama 불필요. RAGSystem/RAGResourceManager 를 모킹합니다.
"""

import io
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
    api_key = "sk_admin_test_token_12345"
    from src.api.api_server import TEST_USER, auth_manager

    auth_manager.register_fixed_api_key(TEST_USER, api_key)
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def mock_rag_resources():
    """RAG 리소스(LLM, Embedder 등)를 모킹하여 무거운 로딩 방지"""
    with patch("src.api.api_server.RAGResourceManager") as mock_mgr:
        mock_llm = MagicMock()
        mock_mgr.get_llm = AsyncMock(return_value=mock_llm)

        mock_embedder = MagicMock()
        mock_embedder.model_name = "mock-embedding-model"
        mock_mgr.get_embedder = AsyncMock(return_value=mock_embedder)

        yield mock_mgr


@pytest.fixture
def mock_rag_system():
    """RAGSystem 클래스 전체 모킹 (build_pipeline/aquery/astream)."""
    with patch("src.api.api_server.RAGSystem") as mock_cls:
        instance = mock_cls.return_value
        instance.build_pipeline = AsyncMock(return_value=("인덱싱 완료", False))
        instance.aquery = AsyncMock(
            return_value={"response": "mock answer", "relevant_docs": []}
        )

        async def astream_impl(query: str, model_name: str | None = None):
            async def _stream():
                yield ("custom", {"content": "Hello"})
                yield ("custom", {"content": "World"})
                yield ("custom", {"documents": []})

            return _stream()

        instance.astream = AsyncMock(side_effect=astream_impl)

        yield mock_cls


@pytest.fixture
def mock_session_manager():
    """SessionManager 상태 모킹 (업로드/질의 사전 체크 통과용)."""
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


@pytest.fixture
def pdf_storage(tmp_path):
    """PDF_STORAGE_DIR을 임시 디렉터리로 패치하여 실제 저장소 오염 방지."""
    with patch("src.api.api_server.PDF_STORAGE_DIR", str(tmp_path)):
        yield tmp_path


@pytest_asyncio.fixture
async def async_client():
    """FastAPI 앱에 연결된 비동기 클라이언트 생성"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- Tests ---


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(
    async_client,
    mock_rag_resources,
    mock_rag_system,
    mock_session_manager,
    auth_headers,
    pdf_storage,
    monkeypatch,
):
    """파일 크기 상한(1024B)을 초과하는 업로드는 413으로 거부되어야 합니다."""
    monkeypatch.setattr(api_server, "MAX_PDF_SIZE_BYTES", 1024)
    files = {"file": ("big.pdf", io.BytesIO(b"x" * 2048), "application/pdf")}
    resp = await async_client.post("/api/v1/upload", files=files, headers=auth_headers)

    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "50MB" in detail or "초과" in detail


@pytest.mark.asyncio
async def test_upload_within_limit_succeeds(
    async_client,
    mock_rag_resources,
    mock_rag_system,
    mock_session_manager,
    auth_headers,
    pdf_storage,
):
    """상한 이내의 정상 업로드는 200으로 성공해야 합니다 (잠금 가드)."""
    files = {"file": ("small.pdf", io.BytesIO(b"%PDF-1.4 small"), "application/pdf")}
    resp = await async_client.post("/api/v1/upload", files=files, headers=auth_headers)

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_query_rejects_overlong_query(
    async_client,
    mock_rag_resources,
    mock_rag_system,
    mock_session_manager,
    auth_headers,
):
    """질문 길이(4000자)를 초과하는 요청은 400으로 거부되어야 합니다."""
    resp = await async_client.post(
        "/api/v1/query",
        json={"query": "q" * 5000, "session_id": "b1"},
        headers=auth_headers,
    )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_stream_query_rejects_overlong_query(
    async_client,
    mock_rag_resources,
    mock_rag_system,
    mock_session_manager,
    auth_headers,
):
    """스트리밍 질의에서 길이 초과 질문은 스트리밍 대신 400으로 거부되어야 합니다."""
    async with async_client.stream(
        "POST",
        "/api/v1/stream_query",
        json={"query": "q" * 5000, "session_id": "b2"},
        headers=auth_headers,
    ) as resp:
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_overlong_session_id(
    async_client,
    mock_rag_resources,
    mock_rag_system,
    mock_session_manager,
    auth_headers,
    pdf_storage,
):
    """업로드 시 64자를 초과하는 세션 ID는 400으로 거부되어야 합니다."""
    files = {"file": ("ok.pdf", io.BytesIO(b"%PDF-1.4 ok"), "application/pdf")}
    resp = await async_client.post(
        "/api/v1/upload",
        files=files,
        data={"session_id": "s" * 100},
        headers=auth_headers,
    )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_session_rejects_overlong_session_id(async_client, auth_headers):
    """세션 삭제 시 64자를 초과하는 세션 ID는 400으로 거부되어야 합니다."""
    resp = await async_client.delete(
        f"/api/v1/session/{'s' * 100}", headers=auth_headers
    )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_query_accepts_normal_length_query(
    async_client,
    mock_rag_resources,
    mock_rag_system,
    mock_session_manager,
    auth_headers,
):
    """정상 길이 질문은 400이 아니어야 합니다 (잠금 가드)."""
    resp = await async_client.post(
        "/api/v1/query",
        json={"query": "정상 길이 질문", "session_id": "c1"},
        headers=auth_headers,
    )

    assert resp.status_code != 400
