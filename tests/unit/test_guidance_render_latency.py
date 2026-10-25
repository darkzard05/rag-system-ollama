# tests/unit/test_guidance_render_latency.py
import time
from unittest.mock import MagicMock, patch
import pytest
from core.session import SessionManager


def test_guidance_panel_renders_without_build_block_overhead_when_no_doc():
    """RED 테스트: 문서가 없는 초기 진입 시 빌드 패널 검사를 건너뛰고 가이드 패널이 즉시 렌더링되어야 함."""
    sid = "test_guidance_perf_session"
    SessionManager.reset()
    SessionManager.init_session(sid)
    SessionManager.set_session_id(sid)

    with (
        patch("ui.components.chat._render_guidance_panel") as mock_guidance,
        patch("ui.components.chat._render_build_progress_block") as mock_build_block,
        patch(
            "ui.components.doc_analysis.panel.should_mount_fragment", return_value=False
        ),
    ):
        from ui.components.chat import render_chat_messages_area

        # 문서가 없는 상태에서 실행
        render_chat_messages_area()

        # [검증 실패 조건 (As-Is)]: 문서가 없는데도 _render_build_progress_block이 호출됨
        # [목표 조건 (To-Be)]: 문서가 없으면 빌드 블록 호출을 완전히 우회하고 가이드 패널만 즉시 호출해야 함
        assert mock_build_block.call_count == 0, (
            "문서가 없는 초기 상태에서는 _render_build_progress_block이 실행되지 않아야 합니다."
        )
        assert mock_guidance.call_count == 1, "가이드 패널이 즉시 렌더링되어야 합니다."
