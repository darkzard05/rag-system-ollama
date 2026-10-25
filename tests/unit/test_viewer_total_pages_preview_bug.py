"""회귀: 1페이지 미리보기 fast-path가 실제 total_pages 산출을 가로채지 않아야 한다."""

import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import ui.components.viewer as viewer_module  # noqa: E402
from core.session import SessionManager  # noqa: E402


def test_resolve_pdf_state_resolves_real_total_pages_despite_preview(
    tmp_path: Path,
) -> None:
    sid = "test_total_pages_bug_sid"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set_session_id(sid)

    pdf_path = tmp_path / "multi_page.pdf"
    doc = pymupdf.open()
    for i in range(5):
        doc.new_page().insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()

    file_hash = "dummy_hash_555"
    SessionManager.set("pdf_file_path", str(pdf_path), sid)
    SessionManager.set("file_hash", file_hash, sid)
    SessionManager.delete("total_pages", sid)

    SessionManager.set(
        "first_page_preview",
        {"file_hash": file_hash, "bytes": b"%PDF-1.4 dummy"},
        sid,
    )

    state = viewer_module._resolve_pdf_state()

    assert state is not None
    assert state["total_pages"] == 5, (
        f"Expected 5 pages, but got {state['total_pages']}"
    )
    assert SessionManager.get("total_pages", session_id=sid) == 5


@pytest.fixture(autouse=True)
def _reset_after() -> None:
    yield
    SessionManager.reset()
