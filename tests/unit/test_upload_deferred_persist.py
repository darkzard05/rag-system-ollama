"""Deferred upload persist — the first frame must not wait on disk I/O.

The upload callback used to write the PDF, delete the previous one and rescan
legacy temp files before the run could produce the viewer. All of that work now
happens in ``_flush_pending_upload``: after the layout (and therefore the
1-page viewer) is rendered, and immediately before the build dispatch.

Contract pinned here:

* the callback publishes viewer state while the target file may not exist yet;
* the flush writes it and only then reclaims the previous file;
* a failed write rolls back to the clean "no document" state;
* pending bytes/hash live in the session SSoT so reruns can still read them;
* ``_handle_pending_tasks`` preflights before dispatching the build;
* the worker flushes before entering ``build_pipeline``.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.main as main_module  # noqa: E402

from core.document_processor import compute_file_hash  # noqa: E402
from core.session import SessionManager  # noqa: E402

SID = "test_deferred_persist"


class FakeSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _make_pdf(text: str = "deferred-persist") -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _wrap_upload(data: bytes, name: str) -> io.BytesIO:
    uploaded = io.BytesIO(data)
    uploaded.name = name  # type: ignore[attr-defined]
    uploaded.type = "application/pdf"  # type: ignore[attr-defined]
    uploaded.size = len(data)  # type: ignore[attr-defined]
    return uploaded


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    yield
    SessionManager.reset()


def _upload_into(tmp_dir: str, data: bytes, name: str = "doc.pdf") -> FakeSessionState:
    fake_state = FakeSessionState({"pdf_uploader": _wrap_upload(data, name)})
    with (
        patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir),
        patch("src.main.st.session_state", fake_state),
        patch("services.monitoring.notification_system.SystemNotifier.success"),
    ):
        main_module.on_file_upload()
    return fake_state


def test_callback_publishes_viewer_state_before_the_file_exists():
    data = _make_pdf()
    with tempfile.TemporaryDirectory() as tmp_dir:
        _upload_into(tmp_dir, data)

        target = Path(tmp_dir) / f"upload_{compute_file_hash('', data=data)}.pdf"
        assert not target.exists(), "콜백은 디스크에 쓰지 않아야 함"
        assert SessionManager.get("pdf_file_path", session_id=SID) == str(target)
        preview = SessionManager.get("first_page_preview", session_id=SID)
        assert preview is not None
        assert preview["bytes"]
        assert SessionManager.get("pending_upload_persist", session_id=SID)

        assert (
            main_module._flush_pending_upload(SID) is main_module.FlushOutcome.PERSISTED
        )
        assert target.exists()
        assert SessionManager.get("pending_upload_persist", session_id=SID) is None


def test_flush_removes_previous_file_only_after_the_new_one_is_written():
    old_dir = tempfile.mkdtemp()
    old_path = Path(old_dir) / "upload_previous.pdf"
    old_path.write_bytes(_make_pdf("previous"))

    data = _make_pdf("new document")
    with tempfile.TemporaryDirectory() as tmp_dir:
        SessionManager.set("pdf_file_path", str(old_path), session_id=SID)
        SessionManager.set("last_uploaded_file_name", "previous.pdf", session_id=SID)
        SessionManager.set("file_hash", "previous-hash", session_id=SID)

        _upload_into(tmp_dir, data, "new.pdf")

        target = Path(tmp_dir) / f"upload_{compute_file_hash('', data=data)}.pdf"
        assert old_path.exists(), "새 파일 기록 전에 이전 파일을 지우면 안 됨"

        main_module._flush_pending_upload(SID)

        assert target.exists()
        assert not old_path.exists()


def test_failed_write_rolls_back_to_the_no_document_state():
    data = _make_pdf()
    with tempfile.TemporaryDirectory() as tmp_dir:
        _upload_into(tmp_dir, data)

        with patch(
            "src.main._persist_upload_atomically", side_effect=OSError("disk full")
        ):
            assert (
                main_module._flush_pending_upload(SID)
                is main_module.FlushOutcome.IO_ERROR
            )

        assert SessionManager.get("pdf_file_path", session_id=SID) is None
        assert SessionManager.get("new_file_uploaded", session_id=SID) is False
        assert SessionManager.get("is_building_rag", session_id=SID) is False
        assert SessionManager.get_build_state(session_id=SID) is None
        assert not list(Path(tmp_dir).glob("upload_*.pdf"))

        messages = SessionManager.get("messages", session_id=SID) or []
        assert any(m.get("msg_type") == "build_error" for m in messages), (
            "저장 실패가 타임라인 메시지로 보고되어야 함"
        )


def test_missing_pending_bytes_rolls_back_and_does_not_dispatch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "upload_missing.pdf"
        SessionManager.set("pdf_file_path", str(target), session_id=SID)
        SessionManager.set("last_uploaded_file_name", "doc.pdf", session_id=SID)
        SessionManager.set("file_hash", "missing-hash", session_id=SID)
        SessionManager.set("is_building_rag", True, session_id=SID)
        SessionManager.init_build_state("doc.pdf", "missing-hash", session_id=SID)
        SessionManager.set(
            "pending_upload_persist",
            {
                "target_path": str(target),
                "needs_write": True,
                "old_path": None,
                "file_name": "doc.pdf",
                "file_hash": "missing-hash",
                "file_bytes": None,
            },
            session_id=SID,
        )
        SessionManager.set("new_file_uploaded", True, session_id=SID)

        built = MagicMock()
        with (
            patch("src.main._bg_rebuild_task", return_value=built) as mock_task,
            patch("src.main.st.rerun"),
        ):
            main_module._handle_pending_tasks()

        mock_task.assert_not_called()
        assert not target.exists()
        assert SessionManager.get("new_file_uploaded", session_id=SID) is False
        assert SessionManager.get("is_building_rag", session_id=SID) is False
        assert SessionManager.get_build_state(session_id=SID) is None


def test_stale_pending_upload_is_not_written_or_deleted():
    data = _make_pdf("stale")
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "upload_stale.pdf"
        pending = {
            "target_path": str(target),
            "needs_write": True,
            "old_path": None,
            "file_name": "old.pdf",
            "file_hash": "old-hash",
            "file_bytes": data,
        }
        SessionManager.set("pending_upload_persist", pending, session_id=SID)
        SessionManager.set("file_hash", "new-hash", session_id=SID)

        outcome = main_module._flush_pending_upload(SID, expected_file_hash="new-hash")

        assert outcome is main_module.FlushOutcome.STALE
        assert not target.exists()
        assert SessionManager.get("pending_upload_persist", session_id=SID) == pending


def test_reset_for_new_file_clears_pending_upload():
    SessionManager.set(
        "pending_upload_persist",
        {"target_path": "x", "needs_write": True},
        session_id=SID,
    )

    SessionManager.reset_for_new_file(SID)

    assert SessionManager.get("pending_upload_persist", session_id=SID) is None


def test_pending_tasks_dispatches_without_blocking_on_disk_write():
    data = _make_pdf()
    with tempfile.TemporaryDirectory() as tmp_dir:
        _upload_into(tmp_dir, data)
        target = Path(tmp_dir) / f"upload_{compute_file_hash('', data=data)}.pdf"
        observed: dict[str, object] = {}

        def fake_worker(coro, session_id):
            observed["session_id"] = session_id
            observed["target_exists"] = target.exists()

        built = MagicMock()
        with (
            patch("src.main._bg_rebuild_task", return_value=built) as mock_task,
            patch("common.utils.run_in_background_worker", side_effect=fake_worker),
            patch("src.main.st.rerun"),
        ):
            main_module._handle_pending_tasks()

        mock_task.assert_called_once()
        assert observed["session_id"] == SID
        assert observed["target_exists"] is False, (
            "메인 스크립트 스레드는 디스크 쓰기를 기다리지 않아야 함"
        )
        assert (
            mock_task.call_args.kwargs["defer_seconds"]
            == main_module._BUILD_START_DELAY_S
        )
        assert mock_task.call_args.kwargs["defer_seconds"] > 0


def test_bg_rebuild_flushes_pending_upload_before_build():
    data = _make_pdf("bg-flush")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _upload_into(tmp_dir, data)
        content_hash = compute_file_hash("", data=data)
        target = Path(tmp_dir) / f"upload_{content_hash}.pdf"
        observed: dict[str, object] = {}

        async def fake_build_pipeline(*args, **kwargs):
            observed["exists"] = Path(kwargs["file_path"]).exists()
            return ("ok", False)

        async def fake_embedder(model_name=None):
            return object()

        with (
            patch("core.model_loader.ModelManager.get_embedder", new=fake_embedder),
            patch("core.rag_core.RAGSystem.build_pipeline", new=fake_build_pipeline),
        ):
            asyncio.run(
                main_module._bg_rebuild_task(
                    SID,
                    str(target),
                    "doc.pdf",
                    "test-embedder",
                    expected_file_hash=content_hash,
                )
            )

        assert observed["exists"] is True, "build_pipeline 진입 시 파일이 있어야 함"
        assert target.exists()


def test_cleanup_orphan_uploads_respects_ttl_and_references():
    with tempfile.TemporaryDirectory() as tmp_dir:
        stale = Path(tmp_dir) / "upload_stale.pdf"
        fresh = Path(tmp_dir) / "upload_fresh.pdf"
        referenced = Path(tmp_dir) / "upload_referenced.pdf"
        for p in (stale, fresh, referenced):
            p.write_bytes(b"%PDF-1.4 orphan")

        old_mtime = time.time() - main_module.UPLOAD_ORPHAN_TTL_S - 60
        os.utime(stale, (old_mtime, old_mtime))
        os.utime(referenced, (old_mtime, old_mtime))

        SessionManager.set("pdf_file_path", str(referenced), session_id=SID)
        with patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir):
            removed = main_module._cleanup_orphan_uploads()

        assert removed == 1
        assert not stale.exists()
        assert fresh.exists(), "TTL 이내 파일은 건드리지 않아야 함"
        assert referenced.exists(), "살아 있는 세션의 파일은 건드리지 않아야 함"
