"""A failed analysis must not leave a stale error bubble behind on retry."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.main as main_module  # noqa: E402

from core.build_state import initial_build_state  # noqa: E402
from core.session import SessionManager  # noqa: E402
from ui.components.doc_analysis.actions import on_retry  # noqa: E402

SID = "test-retry-error-bubble"


@pytest.fixture(autouse=True)
def _clean_session():
    SessionManager.reset_all_state(SID)
    main_module._BUILD_RERUN_LAST.clear()
    yield
    SessionManager.reset_all_state(SID)
    main_module._BUILD_RERUN_LAST.clear()


@pytest.mark.asyncio
async def test_build_failure_error_reuses_build_message_id() -> None:
    SessionManager.set(
        "build_state",
        initial_build_state("doc.pdf", "fail_hash"),
        session_id=SID,
    )
    fake_rag = MagicMock()
    fake_rag.build_pipeline = AsyncMock(side_effect=RuntimeError("extraction failed"))

    with (
        patch(
            "core.model_loader.ModelManager.get_embedder",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch("core.rag_core.RAGSystem", return_value=fake_rag),
    ):
        await main_module._bg_rebuild_task(
            SID,
            "/tmp/doc.pdf",
            "doc.pdf",
            "fake-embedder",
            expected_file_hash="fail_hash",
        )

    errors = [
        m
        for m in SessionManager.get_messages(session_id=SID)
        if m.get("msg_type") == "build_error" and m.get("role") == "assistant"
    ]
    assert len(errors) == 1
    assert errors[0]["msg_id"] == f"build_{SID}"


def test_retry_keeps_unrelated_error_bubbles() -> None:
    SessionManager.add_message(
        "assistant", "Model swap failed", msg_type="build_error", session_id=SID
    )

    on_retry(SID)

    contents = [m["content"] for m in SessionManager.get_messages(session_id=SID)]
    assert "Model swap failed" in contents


def test_next_progress_replaces_the_error_bubble() -> None:
    SessionManager.add_message(
        "assistant",
        "Analysis failed",
        msg_type="build_error",
        msg_id=f"build_{SID}",
        session_id=SID,
    )

    SessionManager.add_message(
        "system",
        "Preparing analysis...",
        msg_type="build_progress",
        msg_id=f"build_{SID}",
        session_id=SID,
    )

    contents = [m["content"] for m in SessionManager.get_messages(session_id=SID)]
    assert contents == ["Preparing analysis..."]
