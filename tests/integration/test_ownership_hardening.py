"""
F2: 소유권 강화 (Ownership Hardening) 통합 테스트.

파일 소유권은 fail-closed(미등록/타인 소유 파일 → 403), 세션 소유권도
fail-closed(미등록 세션은 최초 인증 사용자가 점유하며 타인 접근은 403)입니다.

- test_unbound_existing_file_denied: 파일 fail-closed 전환 검증 (RED 대상)
- test_missing_file_404_for_unbound_hash: resolve-first 404 순서 고정 (regression lock)
- test_concurrent_upload_second_rejected: TOCTOU 경합 검증 (RED 대상)

스윕 로직(_sweep_stale_owners) 검증은 tests/unit/test_ownership.py 로 이관되었습니다.
"""

import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.api_server import app, verify_token

# 테스트용 최소 PDF 콘텐츠 (FileResponse가 정상 응답하도록 충분)
_FAKE_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"


@pytest.fixture(autouse=True)
def auth_override():
    """이 모듈의 테스트에만 인증 우회를 적용하고 이후에는 제거합니다.

    concurrency 테스트에서 사용자 전환이 가능하도록 dict 홀더를 사용합니다.
    """
    holder = {"user_id": "test-user"}

    async def _override_verify_token():
        return holder["user_id"]

    app.dependency_overrides[verify_token] = _override_verify_token
    yield holder
    app.dependency_overrides.pop(verify_token, None)


@pytest.fixture(autouse=True)
def _reset_ownership_registry():
    """모듈 레벨 소유권 레지스트리를 테스트 간 격리합니다."""
    yield
    import src.api.api_server as api

    api._session_owners.clear()
    api._file_owners.clear()


@pytest.fixture
async def client():
    """FastAPI 앱에 연결된 비동기 클라이언트 생성"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def pdf_override(tmp_path: Path):
    """PDF_STORAGE_DIR 을 임시 디렉터리로 패치합니다."""
    with patch("src.api.api_server.PDF_STORAGE_DIR", str(tmp_path)):
        yield tmp_path


@pytest.mark.asyncio
async def test_unbound_existing_file_denied(client, pdf_override):
    """디스크에 존재하지만 소유권에 바인딩되지 않은 파일은 403 으로 거부되어야 합니다 (fail-closed)."""
    fake_hash = "a" * 64
    (Path(pdf_override) / f"{fake_hash}.pdf").write_bytes(_FAKE_PDF)
    resp = await client.get(f"/api/v1/pdf/{fake_hash}")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_missing_file_404_for_unbound_hash(client, pdf_override):
    """존재하지 않는 파일은 resolve-first 순서로 404 를 반환해야 합니다 (unbound 여부와 무관)."""
    resp = await client.get(f"/api/v1/pdf/{'b' * 64}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_concurrent_upload_second_rejected(client, pdf_override, auth_override):
    """첫 업로드가 진행 중일 때 다른 사용자의 동일 세션 업로드는 403 으로 거부되어야 합니다 (TOCTOU)."""

    started = asyncio.Event()
    release = asyncio.Event()
    build_count = 0

    async def _build_pipeline(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            started.set()
            await release.wait()
        return ("인덱싱 완료", False)

    files = {"file": ("test.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")}

    # Patch on the defining class via the bare ``core`` module path:
    # ``get_embedder_for_session`` lives on ``ResourceCoordinator``
    # (src/core/resource_manager.py:561) and the app imports ``core`` (not
    # ``src.core``), so patching ``src.core.resource_manager.ResourceManager...``
    # is a no-op (separate module object) that lets a real Ollama embedder build
    # through. Patch on ``core.resource_manager.ResourceCoordinator`` to intercept.
    with (
        patch("src.api.api_server.RAGSystem.build_pipeline", new=_build_pipeline),
        patch(
            "core.resource_manager.ResourceCoordinator.get_embedder_for_session",
            new_callable=AsyncMock,
        ),
        # Pin path: use_embedder(model_name=...) lazily imports and builds the
        # real embedding model; stub it so the concurrent-upload test never
        # touches Ollama. The pin is a no-op for the injected key.
        patch(
            "core.model_loader.load_embedding_model",
            return_value=MagicMock(),
        ),
    ):
        auth_override["user_id"] = "user-a"
        task_a = asyncio.create_task(
            client.post(
                "/api/v1/upload", files=files, data={"session_id": "race-session"}
            )
        )
        await asyncio.wait_for(started.wait(), timeout=10)
        assert started.is_set(), "first upload never started building pipeline"
        assert build_count == 1

        auth_override["user_id"] = "user-b"
        resp_b = await client.post(
            "/api/v1/upload", files=files, data={"session_id": "race-session"}
        )
        assert resp_b.status_code == 403

        release.set()
        resp_a = await task_a
        assert resp_a.status_code == 200
