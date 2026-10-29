"""Build worker generation guard and upload failure rollback tests."""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.main as main_module  # noqa: E402

from core.build_state import (  # noqa: E402
    BuildPhase,
    initial_build_state,
    is_terminal,
)
from core.session import SessionManager  # noqa: E402

SID = "test_worker_generation_sid"


class FakeSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _make_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "PAGE 1")
    doc.new_page().insert_text((72, 72), "PAGE 2")
    data = doc.tobytes()
    doc.close()
    return data


def _uploaded(pdf_bytes: bytes, name: str = "doc.pdf") -> io.BytesIO:
    uploaded = io.BytesIO(pdf_bytes)
    uploaded.name = name
    uploaded.type = "application/pdf"  # type: ignore[attr-defined]
    uploaded.size = len(pdf_bytes)  # type: ignore[attr-defined]
    return uploaded


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    main_module._BUILD_RERUN_LAST.clear()
    yield
    SessionManager.reset()
    main_module._BUILD_RERUN_LAST.clear()


@pytest.mark.asyncio
async def test_stale_worker_does_not_finalize_newer_build():
    """An old worker must not complete/clear the newer BuildState."""
    old = initial_build_state("old.pdf", "old_hash")
    SessionManager.set("build_state", old, session_id=SID)
    SessionManager.set("is_building_rag", True, session_id=SID)

    new_build_holder: dict[str, object] = {}

    async def fake_build_pipeline(**kwargs):
        new_build = initial_build_state("new.pdf", "new_hash")
        new_build_holder["state"] = new_build
        SessionManager.set("build_state", new_build, session_id=SID)
        SessionManager.set("is_building_rag", True, session_id=SID)
        return "new build done", False

    fake_rag = MagicMock()
    fake_rag.build_pipeline = AsyncMock(side_effect=fake_build_pipeline)

    with (
        patch(
            "core.model_loader.ModelManager.get_embedder",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch("core.rag_core.RAGSystem", return_value=fake_rag),
    ):
        await main_module._bg_rebuild_task(
            SID,
            "/tmp/old.pdf",
            "old.pdf",
            "fake-embedder",
            expected_file_hash="old_hash",
        )

    current = SessionManager.get_build_state(session_id=SID)
    assert current is not None
    assert current.file_hash == "new_hash"
    assert current.phase == BuildPhase.PREPARING
    assert not is_terminal(current.phase)
    assert SessionManager.get("is_building_rag", False, session_id=SID) is True
    assert SessionManager.get("pdf_processed", False, session_id=SID) is False


def test_stale_worker_start_is_ignored_when_hash_already_changed():
    """If the hash changed before the worker body runs, it must not touch flags."""
    new_build = initial_build_state("new.pdf", "new_hash")
    SessionManager.set("build_state", new_build, session_id=SID)
    SessionManager.set("is_building_rag", True, session_id=SID)
    SessionManager.set("rebuild_error", "keep-me", session_id=SID)

    import asyncio

    asyncio.run(
        main_module._bg_rebuild_task(
            SID,
            "/tmp/old.pdf",
            "old.pdf",
            "fake-embedder",
            expected_file_hash="old_hash",
        )
    )

    current = SessionManager.get_build_state(session_id=SID)
    assert current is not None
    assert current.file_hash == "new_hash"
    assert current.phase == BuildPhase.PREPARING
    assert SessionManager.get("is_building_rag", False, session_id=SID) is True
    assert SessionManager.get("rebuild_error", session_id=SID) == "keep-me"


def test_upload_is_rejected_while_build_is_active():
    """A programmatic upload must not reset an in-flight build state."""
    build = initial_build_state("active.pdf", "active_hash")
    SessionManager.set("build_state", build, session_id=SID)
    SessionManager.set("is_building_rag", True, session_id=SID)
    SessionManager.set("new_file_uploaded", False, session_id=SID)

    uploaded = _uploaded(_make_pdf(), "attack.pdf")
    with (
        patch(
            "src.main.st.session_state", FakeSessionState({"pdf_uploader": uploaded})
        ),
        patch("src.main._post_upload_error") as mock_error,
    ):
        main_module.on_file_upload()

    mock_error.assert_called_once()
    current = SessionManager.get_build_state(session_id=SID)
    assert current is not None
    assert current.file_hash == "active_hash"
    assert SessionManager.get("new_file_uploaded", False, session_id=SID) is False


def test_persist_failure_leaves_clean_no_document_state():
    """A disk write failure must not leave PREPARING/is_building stranded."""
    pdf_bytes = _make_pdf()
    uploaded = _uploaded(pdf_bytes, "fail.pdf")

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir),
        patch(
            "src.main.st.session_state", FakeSessionState({"pdf_uploader": uploaded})
        ),
        patch("src.main._persist_upload_atomically", side_effect=OSError("disk full")),
        patch("src.main._post_upload_error") as mock_error,
        patch("services.monitoring.notification_system.SystemNotifier.success"),
    ):
        main_module.on_file_upload()
        main_module._flush_pending_upload(SID)

    mock_error.assert_called_once()
    assert SessionManager.get_build_state(session_id=SID) is None
    assert SessionManager.get("is_building_rag", False, session_id=SID) is False
    assert SessionManager.get("new_file_uploaded", False, session_id=SID) is False
    assert SessionManager.get("pdf_file_path", session_id=SID) is None
    assert SessionManager.get("total_pages", session_id=SID) is None


def test_dispatch_failure_marks_build_error_and_clears_building():
    """A dispatch exception must fail the card, not leave it running forever."""
    SessionManager.set("new_file_uploaded", True, session_id=SID)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", session_id=SID)
    SessionManager.set("file_hash", "hash_dispatch", session_id=SID)
    SessionManager.set("pdf_file_path", "/tmp/doc.pdf", session_id=SID)
    SessionManager.init_build_state("doc.pdf", "hash_dispatch", session_id=SID)

    with (
        patch("src.main._bg_rebuild_task", MagicMock()),
        patch(
            "common.utils.run_in_background_worker",
            side_effect=RuntimeError("submit failed"),
        ),
        patch("src.main.st.rerun") as mock_rerun,
    ):
        main_module._handle_pending_tasks()

    mock_rerun.assert_not_called()
    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    assert build.phase == BuildPhase.ERROR
    assert "dispatch failed" in (build.error or "")
    assert SessionManager.get("is_building_rag", False, session_id=SID) is False
    assert "dispatch failed" in (
        SessionManager.get("pdf_processing_error", "", session_id=SID) or ""
    )


def test_reset_for_new_file_clears_stale_viewer_and_stream_keys():
    SessionManager.set("pdf_file_path", "/tmp/old.pdf", session_id=SID)
    SessionManager.set("total_pages", 9, session_id=SID)
    SessionManager.set("doc_stats", {"cache_used": True}, session_id=SID)
    SessionManager.set("active_stream_msg_id", "old-msg", session_id=SID)
    SessionManager.set("perf_turn_start", 123.0, session_id=SID)
    SessionManager.set("new_file_uploaded", True, session_id=SID)

    SessionManager.reset_for_new_file(session_id=SID)

    assert SessionManager.get("pdf_file_path", session_id=SID) is None
    assert SessionManager.get("total_pages", session_id=SID) is None
    assert SessionManager.get("doc_stats", session_id=SID) is None
    assert SessionManager.get("active_stream_msg_id", session_id=SID) == ""
    assert SessionManager.get("perf_turn_start", session_id=SID) is None
    assert SessionManager.get("new_file_uploaded", False, session_id=SID) is False
