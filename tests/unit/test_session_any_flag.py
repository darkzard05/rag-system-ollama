"""
Unit tests for SessionManager.any_session_flag and background warmup/preload yield guards.
"""

from __future__ import annotations

import pytest

from core.session import SessionManager


def test_any_session_flag_normal_and_edge():
    SessionManager.reset()

    # 1. Normal case: no active flags across any session
    assert (
        SessionManager.any_session_flag("new_file_uploaded", "is_building_rag") is False
    )

    # Set background default session_id
    SessionManager.set_session_id("default")

    # 2. Edge case: user-uuid session has is_building_rag=True, while background thread is on "default"
    SessionManager.set("is_building_rag", True, session_id="user-uuid-1234")

    assert (
        SessionManager.any_session_flag("new_file_uploaded", "is_building_rag") is True
    )
    assert (
        SessionManager.any_session_flag("is_generating_answer", "pdf_file_path")
        is False
    )

    # 3. Test multiple keys checking
    assert (
        SessionManager.any_session_flag(
            "new_file_uploaded",
            "is_building_rag",
            "pdf_file_path",
            "is_generating_answer",
        )
        is True
    )

    SessionManager.reset()


@pytest.mark.asyncio
async def test_loaders_warmup_yield_to_any_session_flag():
    SessionManager.reset()
    SessionManager.set_session_id("default")

    # Set flag in another session
    SessionManager.set("is_building_rag", True, session_id="user-uuid-9999")

    from core._loaders import _warmup_models

    # Should yield immediately without throwing or hanging
    await _warmup_models()


@pytest.mark.asyncio
async def test_pipeline_preload_yield_to_any_session_flag():
    SessionManager.reset()
    SessionManager.set_session_id("default")

    SessionManager.set("is_generating_answer", True, session_id="user-uuid-8888")

    from core.pipeline_builder import _preload_model

    # Should yield immediately in _user_busy check
    await _preload_model()
