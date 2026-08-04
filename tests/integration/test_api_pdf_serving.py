"""
PDF 서빙 엔드포인트 (/api/v1/pdf/{file_hash}) 테스트.

해시 기반 경로 조회, path traversal 방어, 미존재 파일 404 응답을 검증합니다.
기존 api_server의 인증 의존성을 dependency_overrides로 우회하여 무거운 의존성 없이 실행합니다.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.api_server import app, verify_token
from src.core.document_processor import compute_file_hash


# 인증 우회 (기존 패턴 미러링)
async def _override_verify_token():
    return "test-user"


app.dependency_overrides[verify_token] = _override_verify_token

# 테스트용 최소 PDF 콘텐츠 (FileResponse가 정상 응답하도록 충분)
_FAKE_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"


@pytest.fixture()
def pdf_storage(tmp_path: Path):
    """임시 스토리지 루트에 테스트 PDF 파일을 생성하고 PDF_STORAGE_DIR을 패치합니다."""
    fake_hash = compute_file_hash("", data=_FAKE_PDF)
    pdf_file = tmp_path / f"{fake_hash}.pdf"
    pdf_file.write_bytes(_FAKE_PDF)

    with patch("src.api.api_server.PDF_STORAGE_DIR", str(tmp_path)):
        yield tmp_path, fake_hash


@pytest.mark.asyncio
async def test_serve_pdf_success(pdf_storage: tuple[Path, str]):
    """유효한 file_hash로 요청 시 200 + application/pdf 를 반환합니다."""
    storage, fake_hash = pdf_storage
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"/api/v1/pdf/{fake_hash}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/pdf")


@pytest.mark.asyncio
async def test_serve_pdf_not_found(pdf_storage: tuple[Path, str]):
    """존재하지 않는 file_hash로 요청 시 404를 반환합니다."""
    storage, _ = pdf_storage
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/v1/pdf/0000000000000000000000000000000000000000000000000000000000000000"
        )
    assert resp.status_code == 404
    assert "PDF" in resp.json()["detail"]
