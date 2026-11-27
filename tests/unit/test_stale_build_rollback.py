"""A stale build worker must not roll back a newer upload's session state."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.pipeline_builder import PipelineBuilder
from core.session import SessionManager


@pytest.mark.asyncio
async def test_stale_rollback_preserves_newer_upload() -> None:
    sid = "stale-build-rollback"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set("file_hash", "old_hash", session_id=sid)
    SessionManager.set("pdf_file_path", "/tmp/old.pdf", session_id=sid)
    SessionManager.set("last_uploaded_file_name", "old.pdf", session_id=sid)
    SessionManager.set("is_building_rag", True, session_id=sid)

    async def fail_after_new_upload(*_args: object, **_kwargs: object) -> None:
        SessionManager.set("file_hash", "new_hash", session_id=sid)
        SessionManager.set("pdf_file_path", "/tmp/new.pdf", session_id=sid)
        SessionManager.set("last_uploaded_file_name", "new.pdf", session_id=sid)
        raise RuntimeError("old build failed")

    builder = PipelineBuilder(session_id=sid)
    with (
        patch.object(builder, "_build_impl", side_effect=fail_after_new_upload),
        pytest.raises(RuntimeError, match="old build failed"),
    ):
        await builder.build(
            file_path="/tmp/old.pdf",
            file_name="old.pdf",
            embedder=MagicMock(),
            file_hash="old_hash",
        )

    assert SessionManager.get("file_hash", session_id=sid) == "new_hash"
    assert SessionManager.get("pdf_file_path", session_id=sid) == "/tmp/new.pdf"
    assert SessionManager.get("last_uploaded_file_name", session_id=sid) == "new.pdf"
    assert SessionManager.get("is_building_rag", False, session_id=sid) is True
