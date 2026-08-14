"""
PDF 라이브러리 보존 수명주기 테스트 (F3 - TDD).

검증 대상:
1. API 업로드 PDF는 세션 삭제 이후에도 서빙 가능 (pdf_library_path 마커)
2. 업로드 직후 동일 file_hash로 PDF 서빙 가능 (라이브러리 디렉터리 이동 회귀 가드)
3. 보존 기간(_PDF_RETENTION_DAYS)을 초과한 라이브러리 PDF와 소유권 항목 제거
4. 최근 라이브러리 PDF는 스윕에서 보존
5. UI 전용 임시 파일(pdf_file_path만 설정)은 세션 삭제 시 여전히 제거 (스킵 조건 잠금)

테스트 환경: Ollama 불필요. 업로드 시 RAGSystem/get_embedder 를 모킹합니다.
"""

import io
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import src.api.api_server as api
from fastapi.testclient import TestClient
from src.api.api_server import app, verify_token

from core.session import SessionManager

client = TestClient(app)


async def _override_verify_token():
    return "test-user"


@pytest.fixture(autouse=True)
def _auth_override_and_cleanup():
    """동일 사용자('test-user') 인증 우회 + 세션 상태 격리.

    다른 테스트 모듈이 남긴 verify_token 오버라이드가 인증 검증을
    우회하지 않도록 제거하고, 각 테스트 후 세션 상태를 초기화합니다.
    """
    app.dependency_overrides.pop(verify_token, None)
    SessionManager.reset()
    app.dependency_overrides[verify_token] = _override_verify_token
    yield
    app.dependency_overrides.pop(verify_token, None)
    SessionManager.reset()


@pytest.fixture
def pdf_storage(tmp_path: Path) -> Path:
    """PDF_STORAGE_DIR을 임시 디렉터리로 패치합니다.

    업로드 → 세션 삭제 → 서빙이 모두 동일한 스토리지 루트를 사용하도록
    테스트 전체 구간에서 상수를 패치합니다.
    """
    with patch("src.api.api_server.PDF_STORAGE_DIR", str(tmp_path)):
        yield tmp_path


def _upload(session_id: str):
    """무거운 리소스(RAGSystem, embedder)를 모킹하여 업로드를 수행합니다."""
    mock_rag_system = MagicMock()
    mock_rag_system.return_value.build_pipeline = AsyncMock(
        return_value=("인덱싱 완료", False)
    )
    pdf = {"file": ("test.pdf", io.BytesIO(b"%PDF-1.4 mock"), "application/pdf")}
    with (
        patch("src.api.api_server.RAGSystem", mock_rag_system),
        patch(
            "src.core.resource_manager.ResourceManager.get_embedder_for_session",
            new_callable=AsyncMock,
        ),
    ):
        return client.post(
            "/api/v1/upload",
            files=pdf,
            data={"session_id": session_id},
        )


def test_uploaded_pdf_survives_session_deletion(pdf_storage: Path):
    """API 업로드 PDF는 세션 삭제 이후에도 서빙 가능해야 합니다 (라이브러리 보존)."""
    resp = _upload("lib-sess")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    file_hash = data["file_hash"]
    session_id = data["session_id"]

    del_resp = client.delete(f"/api/v1/session/{session_id}")
    assert del_resp.status_code == 200, del_resp.text

    pdf_resp = client.get(f"/api/v1/pdf/{file_hash}")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"].startswith("application/pdf")


def test_uploaded_pdf_servable_after_upload(pdf_storage: Path):
    """업로드 직후 동일 file_hash로 PDF 서빙이 가능해야 합니다 (디렉터리 이동 가드)."""
    resp = _upload("serve-sess")
    assert resp.status_code == 200, resp.text
    file_hash = resp.json()["file_hash"]

    pdf_resp = client.get(f"/api/v1/pdf/{file_hash}")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"].startswith("application/pdf")


def test_library_sweep_removes_expired_files(tmp_path: Path):
    """보존 기간(_PDF_RETENTION_DAYS)을 초과한 라이브러리 PDF와 소유권 항목을 제거합니다."""
    stale_hash = "a" * 64
    stale_file = tmp_path / f"{stale_hash}.pdf"
    stale_file.write_bytes(b"%PDF-1.4 expired")
    old_time = time.time() - 31 * 24 * 3600
    os.utime(stale_file, (old_time, old_time))
    api._bind_file_owner(stale_hash, "user-x")

    with patch("src.api.api_server.PDF_STORAGE_DIR", str(tmp_path)):
        api._sweep_expired_library_files()

    assert not stale_file.exists()
    assert stale_hash not in api._file_owners


def test_library_sweep_keeps_recent_files(tmp_path: Path):
    """최근(보존 기간 내) 라이브러리 PDF와 소유권 항목은 스윕에서 보존됩니다."""
    fresh_hash = "b" * 64
    fresh_file = tmp_path / f"{fresh_hash}.pdf"
    fresh_file.write_bytes(b"%PDF-1.4 fresh")
    api._bind_file_owner(fresh_hash, "user-x")

    with patch("src.api.api_server.PDF_STORAGE_DIR", str(tmp_path)):
        api._sweep_expired_library_files()

    assert fresh_file.exists()
    assert fresh_hash in api._file_owners


def test_ui_temp_file_still_deleted_on_session_delete(tmp_path: Path):
    """UI 전용 임시 파일(pdf_file_path만 설정)은 세션 삭제 시 여전히 제거되어야 합니다."""
    sid = "ui-temp-sess"
    SessionManager.init_session(session_id=sid)
    temp_file = tmp_path / "ui_upload.pdf"
    temp_file.write_bytes(b"%PDF-1.4 mock")
    SessionManager.set("pdf_file_path", str(temp_file), session_id=sid)

    assert SessionManager.delete_session(sid) is True
    assert not temp_file.exists()
