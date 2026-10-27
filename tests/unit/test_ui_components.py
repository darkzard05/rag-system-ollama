import os
import sys
import unittest
import tempfile

from streamlit.testing.v1 import AppTest

from src.core.session import SessionManager  # type: ignore[import]

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class TestChatUI(unittest.TestCase):
    def test_reference_list_generation(self):
        """
        문서별 한 줄 요약 참조 목록이 새 계약대로 생성되는지 검증합니다.
        """
        from ui.components.chat import render_message
        from unittest.mock import MagicMock, patch

        def _doc(page, text):
            d = MagicMock()
            d.metadata = {"page": page}
            d.page_content = text
            return d

        documents = [_doc(i, f"관련 문장 {i}번입니다. " * 10) for i in range(1, 7)]
        metrics = {"doc_count": 6, "total_time": 1.5}

        mock_st = MagicMock()
        with (
            patch("ui.components.chat.st", mock_st),
            patch("ui.components.chat_references.st", mock_st),
        ):
            render_message(
                role="assistant",
                content="테스트 답변입니다.",
                documents=documents,
                metrics=metrics,
                msg_index=0,
                is_latest=True,
            )

        markdown_texts = [
            str(call.args[0]) for call in mock_st.markdown.call_args_list if call.args
        ]
        assert any("Reference" in t for t in markdown_texts), (
            "Reference 見出이 없습니다."
        )

        buttons = mock_st.button.call_args_list
        assert len(buttons) == 5, f"참조 행 상한 위반: {len(buttons)}"
        first_label = str(buttons[0].args[0])
        assert first_label.startswith("1. :blue[")
        assert ":gray[p.1]" in first_label


def test_pdf_viewer_key_includes_page():
    """페이지가 키에 포함되어 페이지 변경 시 컴포넌트 재마운트가 보장된다."""
    from src.ui.widget_keys import pdf_viewer_key  # type: ignore[import]

    assert pdf_viewer_key("abc", 1) == "pdf_v8_abc_1"
    assert pdf_viewer_key("abc", 1) != pdf_viewer_key("abc", 2)
    assert pdf_viewer_key("abc", 1) != pdf_viewer_key("abd", 1)


def test_page_jump_click_invokes_handler_via_callback():
    from ui.components.chat_references import (
        _render_references_content,
        _handle_page_jump,
    )
    from core.session import SessionManager
    from unittest.mock import MagicMock, patch

    SessionManager.reset()
    doc = MagicMock()
    doc.metadata = {"page": 3}

    mock_st = MagicMock()
    with patch("ui.components.chat_references.st", mock_st):
        _render_references_content(
            msg_id="m1",
            documents=[doc],
            citations=None,
            generating=False,
        )

    btn_calls = mock_st.button.call_args_list
    jump_btn = next(call for call in btn_calls if "p.3" in str(call.args[0]))
    assert jump_btn.kwargs.get("on_click") is _handle_page_jump
    assert jump_btn.kwargs.get("args") == (3,)

    fake_state = {}
    with (
        patch("ui.components.chat_references.st.session_state", fake_state),
        patch("ui.components.common.st.session_state", fake_state),
    ):
        _handle_page_jump(3)

    token = SessionManager.get("pdf_target_page")
    assert isinstance(token, dict) and token["page"] == 3
    assert SessionManager.get("current_page") == 3


def test_doc_jump_click_invokes_handler_via_callback():
    from ui.components.chat_references import (
        _render_references_content,
        _handle_page_jump,
    )
    from core.session import SessionManager
    from unittest.mock import MagicMock, patch

    SessionManager.reset()
    doc = MagicMock()
    doc.metadata = {"page": 7, "doc_id": "docA"}

    mock_st = MagicMock()
    with patch("ui.components.chat_references.st", mock_st):
        _render_references_content(
            msg_id="m1",
            documents=[doc],
            citations=[{"doc_id": "docA", "section": "Intro", "page": 7}],
            generating=False,
        )

    btn_calls = mock_st.button.call_args_list
    jump_btn = next(call for call in btn_calls if "Intro · p.7" in str(call.args[0]))
    assert jump_btn.kwargs.get("on_click") is _handle_page_jump
    assert jump_btn.kwargs.get("args") == (7,)

    fake_state = {}
    with (
        patch("ui.components.chat_references.st.session_state", fake_state),
        patch("ui.components.common.st.session_state", fake_state),
    ):
        _handle_page_jump(7)

    token = SessionManager.get("pdf_target_page")
    assert isinstance(token, dict) and token["page"] == 7
    assert SessionManager.get("current_page") == 7


class TestGeneratingExpanderRerender(unittest.TestCase):
    def test_double_render_same_turn_no_duplicate_key(self):
        """회귀: 동일 턴 생성중 익스팬더 2회 렌더(타임라인+aux)도 키 충돌 없음.

        생성 중 인용 버튼에 안정 키(pop_doc_...)를 부여하면 두 컨테이너가
        같은 키를 등록해 StreamlitDuplicateElementKey로 스트리밍이 즉시
        죽는다. 생성 중 버튼은 키 없이(위치 자동 키) 렌더해야 한다.
        """
        script_content = """
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from unittest.mock import MagicMock
from ui.components.chat_references import render_generation_expander

doc = MagicMock()
doc.metadata = {"page": 3}

msg = {
    "thought": "",
    "documents": [doc],
    "citations": [],
    "metrics": {},
    "msg_id": "dup-m1",
}
with st.container():
    render_generation_expander(msg, expanded=False, generating=True)
with st.container():
    render_generation_expander(msg, expanded=False, generating=True)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "temp_test_genexp_dupkey.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            at = AppTest.from_file(script_path).run(timeout=60)
            assert not at.exception, at.exception
