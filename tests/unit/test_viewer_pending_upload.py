"""디지/지연 저장(Defer-Persist) 구간 뷰어 컬럼 메모리 폴백 및 해시 검증 단위 테스트."""

import sys
import types
from pathlib import Path
from typing import Any

import pymupdf
import pytest

import ui.components.viewer as viewer_mod
from core.session import SessionManager
from ui.components.viewer import _get_pdf_page_bytes, render_pdf_viewer


def _make_two_page_bytes() -> bytes:
    """유효한 2페이지 PDF 바이트를 생성한다."""
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)
    doc[0].insert_text((72, 100), "PAGE ONE")
    doc[1].insert_text((72, 200), "PAGE TWO")
    data = doc.tobytes()
    doc.close()
    return data


def test_get_pdf_page_bytes_pending_upload_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """지연 저장 중 디스크 파일이 없을 때 SessionManager SSoT에서 바이트를 정상 폴백하고 페이지를 추출한다."""
    # 1. sys.modules에서 "__main__", "src.main", "main" 모듈 속성 제거/숨김 (Streamlit ScriptRunner 환경 시뮬레이션)
    for mod_name in ("__main__", "src.main", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "get_pending_upload_bytes"):
            monkeypatch.delattr(mod, "get_pending_upload_bytes", raising=False)

    pdf_bytes = _make_two_page_bytes()
    session_id = SessionManager.get_session_id()

    # SessionManager SSoT에 pending_upload_persist 등록
    SessionManager.set(
        "pending_upload_persist",
        {"file_hash": "hash_a", "file_bytes": pdf_bytes},
        session_id=session_id,
    )

    nonexistent_path = str(tmp_path / "nonexistent.pdf")

    # 2페이지 추출 요청
    res = _get_pdf_page_bytes(nonexistent_path, "hash_a", 2)
    assert res
    with pymupdf.open(stream=res, filetype="pdf") as single:
        assert len(single) == 1
        text = single[0].get_text()
        assert "PAGE TWO" in text


def test_render_pdf_viewer_pending_upload_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """지연 저장 중 2페이지 렌더링 시 show_pdf_error("data") 없이 pdf_viewer가 호출되는지 검증."""
    for mod_name in ("__main__", "src.main", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "get_pending_upload_bytes"):
            monkeypatch.delattr(mod, "get_pending_upload_bytes", raising=False)

    pdf_bytes = _make_two_page_bytes()
    session_id = SessionManager.get_session_id()
    SessionManager.set(
        "pending_upload_persist",
        {"file_hash": "hash_a", "file_bytes": pdf_bytes},
        session_id=session_id,
    )

    nonexistent_path = str(tmp_path / "nonexistent.pdf")

    calls: dict = {}
    error_called: list = []

    def stub_viewer(**kwargs: Any) -> None:
        calls.update(kwargs)

    def stub_error(err_type: str) -> None:
        error_called.append(err_type)

    # streamlit_pdf_viewer 모듈 스텁 설정
    try:
        import streamlit_pdf_viewer
    except ImportError:
        module = types.ModuleType("streamlit_pdf_viewer")
        module.__dict__["pdf_viewer"] = stub_viewer
        sys.modules["streamlit_pdf_viewer"] = module
    else:
        monkeypatch.setattr(streamlit_pdf_viewer, "pdf_viewer", stub_viewer)

    monkeypatch.setattr(viewer_mod, "show_pdf_error", stub_error)

    render_pdf_viewer(nonexistent_path, 2, "hash_a")

    assert "data" not in error_called
    assert calls.get("input")
    assert calls.get("pages_to_render") == [1]  # 단일 페이지 추출 후 1페이지로 렌더


def test_get_pdf_page_bytes_stale_hash_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """file_hash가 불일치할 경우(스테일 바이트) 메모리 폴백이 차단되고 b""를 반환한다."""
    for mod_name in ("__main__", "src.main", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "get_pending_upload_bytes"):
            monkeypatch.delattr(mod, "get_pending_upload_bytes", raising=False)

    pdf_bytes = _make_two_page_bytes()
    session_id = SessionManager.get_session_id()
    SessionManager.set(
        "pending_upload_persist",
        {"file_hash": "hash_old", "file_bytes": pdf_bytes},
        session_id=session_id,
    )

    nonexistent_path = str(tmp_path / "nonexistent.pdf")

    # 요청된 해시가 "hash_new"이므로 "hash_old"와 불일치 -> b"" 반환
    res = _get_pdf_page_bytes(nonexistent_path, "hash_new", 2)
    assert res == b""


def test_get_pdf_page_bytes_empty_or_none_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pending_upload_persist가 없거나 비어 있을 때 예외 없이 b""를 반환한다."""
    for mod_name in ("__main__", "src.main", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "get_pending_upload_bytes"):
            monkeypatch.delattr(mod, "get_pending_upload_bytes", raising=False)

    session_id = SessionManager.get_session_id()
    SessionManager.delete("pending_upload_persist", session_id=session_id)

    nonexistent_path = str(tmp_path / "nonexistent.pdf")

    assert _get_pdf_page_bytes(nonexistent_path, "hash_a", 2) == b""
