"""Upload dedup guard — same-bytes re-upload must not mint new files.

Wave-0: data/temp/upload_*.pdf grew unbounded (454 files/2.02GB) because
every upload minted ``upload_{sid}_{epoch}.pdf`` even for identical bytes,
and the reset asymmetry (``file_hash`` cleared, ``last_uploaded_file_name``
kept) defeated the "already uploaded" guard.

All tests use ``tmp_path`` only — never the real ``data/temp``.
"""

import io
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

# 프로젝트 루트를 path에 추가 (test_main_background_tasks.py와 동일 관례).
sys.path.append(os.path.abspath("src"))

from src.main import on_file_upload

from core.document_processor import compute_file_hash
from core.session import SessionManager


class FakeSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _make_test_pdf(text="test"):
    """fitz로 최소 유효 PDF 바이트 생성 (upload 검증 게이트 통과용)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _bind_session():
    sid = f"dedup_{uuid.uuid4().hex[:8]}"
    SessionManager.set_session_id(sid)
    return sid


def _upload(file_bytes, name):
    uploaded = io.BytesIO(file_bytes)
    uploaded.name = name
    uploaded.type = "application/pdf"
    uploaded.size = len(file_bytes)
    return FakeSessionState({"pdf_uploader": uploaded})


def _upload_files(temp_dir):
    return sorted(Path(temp_dir).glob("upload_*.pdf"))


def test_reset_for_new_file_clears_both_guard_keys():
    """가드 비대칭: reset이 file_hash와 파일명을 함께 비워야 한다."""
    sid = _bind_session()
    SessionManager.set("last_uploaded_file_name", "a.pdf", session_id=sid)
    SessionManager.set("file_hash", "hash-a", session_id=sid)

    SessionManager.reset_for_new_file(session_id=sid)

    assert SessionManager.get("file_hash", session_id=sid) is None
    assert SessionManager.get("last_uploaded_file_name", session_id=sid) is None


def test_same_bytes_different_name_reuses_existing_file(tmp_path):
    """동일 바이트 + 다른 파일명 재업로드 → 새 파일 없이 기존 파일 재사용."""
    sid = _bind_session()
    file_bytes = _make_test_pdf("dedup-same-bytes")
    content_hash = compute_file_hash("", data=file_bytes)

    seed = tmp_path / "upload_seed_1.pdf"
    seed.write_bytes(file_bytes)
    SessionManager.set("last_uploaded_file_name", "a.pdf", session_id=sid)
    SessionManager.set("file_hash", content_hash, session_id=sid)
    SessionManager.set("pdf_file_path", str(seed), session_id=sid)

    with (
        patch("src.main.FilePathConstants.TEMP_DIR", str(tmp_path)),
        patch("src.main.st.session_state", _upload(file_bytes, "b.pdf")),
        patch(
            "services.monitoring.notification_system.SystemNotifier.success"
        ) as mock_success,
    ):
        on_file_upload()

    assert len(_upload_files(str(tmp_path))) == 1
    assert SessionManager.get("pdf_file_path", session_id=sid) == str(seed)
    mock_success.assert_called_once()


def test_repeated_same_bytes_uploads_keep_single_file(tmp_path):
    """before/after 카운트 시뮬레이션: 동일 내용 3회 업로드 → 파일 1개 유지."""
    sid = _bind_session()
    file_bytes = _make_test_pdf("count-simulation")

    with (
        patch("src.main.FilePathConstants.TEMP_DIR", str(tmp_path)),
        patch("services.monitoring.notification_system.SystemNotifier.success"),
    ):
        paths = []
        for i in range(3):
            with patch(
                "src.main.st.session_state",
                _upload(file_bytes, f"r{i}.pdf"),
            ):
                on_file_upload()
            paths.append(SessionManager.get("pdf_file_path", session_id=sid))

    assert len(_upload_files(str(tmp_path))) == 1
    # 삭제-재생성 churn 없이 동일 파일이 유지되어야 한다.
    assert paths[0] == paths[1] == paths[2]
    assert paths[0] in {str(p) for p in _upload_files(str(tmp_path))}


def test_same_name_same_hash_creates_no_new_file(tmp_path):
    """완전 동일 재업로드 → temp에 파일이 생기지 않고 guard가 발동한다."""
    sid = _bind_session()
    file_bytes = _make_test_pdf("identical")
    SessionManager.set("last_uploaded_file_name", "same.pdf", session_id=sid)
    SessionManager.set(
        "file_hash", compute_file_hash("", data=file_bytes), session_id=sid
    )

    with (
        patch("src.main.FilePathConstants.TEMP_DIR", str(tmp_path)),
        patch("src.main.st.session_state", _upload(file_bytes, "same.pdf")),
        patch(
            "services.monitoring.notification_system.SystemNotifier.success"
        ) as mock_success,
    ):
        on_file_upload()

    assert _upload_files(str(tmp_path)) == []
    assert not SessionManager.get("new_file_uploaded", session_id=sid)
    mock_success.assert_called_once()
    assert "already uploaded" in mock_success.call_args.args[0]


def test_locked_old_file_delete_is_retried(tmp_path):
    """잠긴 기존 파일 삭제 → 재시도되어 결국 제거되고 업로드는 성공한다."""
    sid = _bind_session()
    old_bytes = _make_test_pdf("old-doc")
    new_bytes = _make_test_pdf("new-doc")

    old_path = tmp_path / "upload_old_1.pdf"
    old_path.write_bytes(old_bytes)
    SessionManager.set("last_uploaded_file_name", "old.pdf", session_id=sid)
    SessionManager.set(
        "file_hash", compute_file_hash("", data=old_bytes), session_id=sid
    )
    SessionManager.set("pdf_file_path", str(old_path), session_id=sid)

    real_remove = os.remove
    calls = []

    def flaky_remove(path, *args, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError("file locked")
        return real_remove(path, *args, **kwargs)

    with (
        patch("src.main.FilePathConstants.TEMP_DIR", str(tmp_path)),
        patch("src.main.st.session_state", _upload(new_bytes, "new.pdf")),
        patch("os.remove", side_effect=flaky_remove),
        patch("services.monitoring.notification_system.SystemNotifier.success"),
    ):
        on_file_upload()

    assert len(calls) >= 2
    assert not old_path.exists()
    assert SessionManager.get("new_file_uploaded", session_id=sid)


def test_unremovable_old_file_is_reported_not_leaked(tmp_path):
    """삭제 불가 기존 파일 → 조용히 누수시키지 않고 warning으로 보고한다."""
    sid = _bind_session()
    old_bytes = _make_test_pdf("locked-old")
    new_bytes = _make_test_pdf("locked-new")

    old_path = tmp_path / "upload_locked_1.pdf"
    old_path.write_bytes(old_bytes)
    SessionManager.set("last_uploaded_file_name", "old.pdf", session_id=sid)
    SessionManager.set(
        "file_hash", compute_file_hash("", data=old_bytes), session_id=sid
    )
    SessionManager.set("pdf_file_path", str(old_path), session_id=sid)

    with (
        patch("src.main.FilePathConstants.TEMP_DIR", str(tmp_path)),
        patch("src.main.st.session_state", _upload(new_bytes, "new.pdf")),
        patch("os.remove", side_effect=PermissionError("locked")),
        patch("services.monitoring.notification_system.SystemNotifier.success"),
        patch(
            "services.monitoring.notification_system.SystemNotifier.warning"
        ) as mock_warning,
    ):
        on_file_upload()

    mock_warning.assert_called_once()
    assert SessionManager.get("new_file_uploaded", session_id=sid)
