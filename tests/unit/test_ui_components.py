import os
import sys
import unittest

from streamlit.testing.v1 import AppTest

from core.session import SessionManager

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class TestChatUI(unittest.TestCase):
    def test_page_jump_button_generation(self):
        """
        참조 점프 버튼이 'p.N' 단일화 라벨로 정상 생성되는지 검증합니다.

        현행 계약 (chat_references._render_references_content):
        - 라벨: 섹션명이 없으면 'p.{page}' (예: 'p.1', 'p.12')
        - 키: 'pop_doc_{msg_id}_{page}_{idx}'
        """
        # 임시 테스트 스크립트 작성
        script_content = """
import streamlit as st
import sys
import os

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from ui.components.chat import render_message
from unittest.mock import MagicMock

# 모의 데이터 설정
msg_index = 0
role = "assistant"
content = "테스트 답변입니다."

# Document 객체 모사 (metadata 속성 포함)
doc1 = MagicMock()
doc1.metadata = {"page": 1}
doc2 = MagicMock()
doc2.metadata = {"page": 12}

documents = [doc1, doc2]
metrics = {"doc_count": 2, "input_token_count": 10, "token_count": 20, "total_time": 1.5, "ttft": 0.1}

# UI 렌더링 호출
render_message(
    role=role,
    content=content,
    documents=documents,
    metrics=metrics,
    msg_index=msg_index,
    is_latest=True,
)
"""
        with open("temp_test_ui.py", "w", encoding="utf-8") as f:
            f.write(script_content)

        try:
            # AppTest를 사용하여 스크립트 실행.
            # 기본 3s 타임아웃은 무거운 임포트 체인 + 팝오버 렌더 경로에
            # 비해 빠듯하여(이 환경에서 재현 확인) 명시적 타임아웃을 부여한다.
            at = AppTest.from_file("temp_test_ui.py").run(timeout=60)

            # 1. 버튼들 확인 ('p.N' 단일화 라벨)
            button_labels = [b.label for b in at.button]

            # 'p.N' 형식 라벨 확인
            assert "p.1" in button_labels, "p.1 버튼이 없습니다."
            assert "p.12" in button_labels, "p.12 버튼이 없습니다."
            # 2. 버튼 키 접두사 확인 (p.12 버튼)
            jump_button_12 = next(b for b in at.button if b.label == "p.12")
            assert jump_button_12.key.startswith("pop_doc_msg_0_12_"), (
                f"버튼 키 오류: {jump_button_12.key}"
            )

            # 3. CSS 제거 확인 (기존 스타일 코드가 없는지)
            style_markdowns = [
                m for m in at.markdown if "white-space: nowrap !important" in m.value
            ]
            assert len(style_markdowns) == 0, "버튼 스타일 CSS가 여전히 존재합니다."

        finally:
            if os.path.exists("temp_test_ui.py"):
                os.remove("temp_test_ui.py")


def test_pdf_viewer_key_includes_page():
    """페이지가 키에 포함되어 페이지 변경 시 컴포넌트 재마운트가 보장된다."""
    from ui.widget_keys import pdf_viewer_key

    assert pdf_viewer_key("abc", 1) == "pdf_v8_abc_1"
    assert pdf_viewer_key("abc", 1) != pdf_viewer_key("abc", 2)
    assert pdf_viewer_key("abc", 1) != pdf_viewer_key("abd", 1)


def test_page_jump_click_invokes_handler_via_callback():
    """P0 회귀: 페이지 점프 버튼 클릭이 on_click 콜백(콜백 페이즈)으로
    pdf_target_page 점프 토큰을 정확히 1회 세팅한다. 렌더 단계 인라인 호출로
    회귀하면 클릭 전부터 토큰이 세팅되어 아래 선행 단언이 실패한다.

    현행 계약 (chat_references._handle_page_jump):
    - SessionManager 'pdf_target_page' = {"page", "source": "manual", "ts"}
    - navigate_to_page로 nav-input('pdf_nav_input_v6') 동기화
    """
    script_content = """
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from unittest.mock import MagicMock
import ui.components.chat_references as refs
from ui.components.chat_references import _render_references_content
from core.session import SessionManager

doc = MagicMock()
doc.metadata = {"page": 3}

# on_click 호출 횟수를 세는 래퍼 (rerun을 넘겨 유지되도록 session_state에 누적).
# 스크립트는 매 rerun마다 재실행되므로 가드 없이 래핑하면 래퍼가 중첩되어
# 1회 클릭이 여러 번 카운트된다 — 1회만 래핑한다.
if not getattr(refs._handle_page_jump, "_is_counting_wrapper", False):
    _true_orig_jump = refs._handle_page_jump

    def _counting_jump(p):
        if "__jump_count" not in st.session_state:
            st.session_state["__jump_count"] = 1
        else:
            st.session_state["__jump_count"] = (
                int(st.session_state["__jump_count"]) + 1
            )
        return _true_orig_jump(p)

    _counting_jump._is_counting_wrapper = True  # type: ignore[attr-defined]
    refs._handle_page_jump = _counting_jump

_render_references_content(
    msg_id="m1",
    documents=[doc],
    citations=None,
    generating=False,
)

# 콜백이 세팅한 점프 토큰을 평탄 키로 노출 (외부 단언용)
_token = SessionManager.get("pdf_target_page")
if isinstance(_token, dict) and "page" in _token:
    st.session_state["__jumped_page"] = int(_token["page"])
"""
    SessionManager.reset()
    with open("temp_test_jump_callback.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    try:
        at = AppTest.from_file("temp_test_jump_callback.py").run(timeout=60)
        assert not at.exception, at.exception
        # 렌더 단계에서는 핸들러가 호출되지 않아야 한다 (P0 회귀 가드)
        assert "__jumped_page" not in at.session_state, (
            "렌더 단계에서 점프 토큰이 세팅됨 (인라인 호출 회귀 의심)"
        )
        jump_btn = next(b for b in at.button if b.label == "p.3")
        assert jump_btn.key.startswith("pop_doc_m1_3_"), jump_btn.key
        at_after = jump_btn.click().run(timeout=60)
        assert not at_after.exception, at_after.exception
        assert "__jump_count" in at_after.session_state
        assert at_after.session_state["__jump_count"] == 1
        assert at_after.session_state["__jumped_page"] == 3
        assert at_after.session_state["pdf_nav_input_v6"] == 3
    finally:
        if os.path.exists("temp_test_jump_callback.py"):
            os.remove("temp_test_jump_callback.py")


def test_doc_jump_click_invokes_handler_via_callback():
    """P0 회귀: 인용(doc) 점프 버튼 클릭이 on_click 콜백으로
    pdf_target_page 토큰(page=인용 메타)을 정확히 1회 세팅한다.

    현행 계약: citations 항목 {"doc_id", "section", "page"}이
    "'{section} · p.{page}'" 버튼으로 렌더되며 클릭 시 _handle_page_jump이
    {"page", "source": "manual", "ts"} 토큰 + nav-input 동기화를 수행한다.
    """
    script_content = """
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from unittest.mock import MagicMock
import ui.components.chat_references as refs
from ui.components.chat_references import _render_references_content
from core.session import SessionManager

doc = MagicMock()
doc.metadata = {"page": 7, "doc_id": "docA"}

if not getattr(refs._handle_page_jump, "_is_counting_wrapper", False):
    _true_orig_jump = refs._handle_page_jump

    def _counting_jump(p):
        if "__jump_count" not in st.session_state:
            st.session_state["__jump_count"] = 1
        else:
            st.session_state["__jump_count"] = (
                int(st.session_state["__jump_count"]) + 1
            )
        return _true_orig_jump(p)

    _counting_jump._is_counting_wrapper = True  # type: ignore[attr-defined]
    refs._handle_page_jump = _counting_jump

_render_references_content(
    msg_id="m1",
    documents=[doc],
    citations=[{"doc_id": "docA", "section": "Intro", "page": 7}],
    generating=False,
)

_token = SessionManager.get("pdf_target_page")
if isinstance(_token, dict) and "page" in _token:
    st.session_state["__jumped_page"] = int(_token["page"])
"""
    SessionManager.reset()
    with open("temp_test_docjump.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    try:
        at = AppTest.from_file("temp_test_docjump.py").run(timeout=60)
        assert not at.exception, at.exception
        assert "__jumped_page" not in at.session_state, (
            "렌더 단계에서 점프 토큰이 세팅됨 (인라인 호출 회귀 의심)"
        )
        btn = next(b for b in at.button if b.label == "Intro · p.7")
        assert btn.key.startswith("pop_doc_m1_7_"), btn.key
        at_after = btn.click().run(timeout=60)
        # P0 계약: 클릭이 on_click 콜백 페이즈에서 _handle_page_jump을 실행한다.
        assert not at_after.exception, at_after.exception
        assert "__jump_count" in at_after.session_state
        assert at_after.session_state["__jump_count"] == 1
        assert at_after.session_state["__jumped_page"] == 7
        assert at_after.session_state["pdf_nav_input_v6"] == 7
    finally:
        if os.path.exists("temp_test_docjump.py"):
            os.remove("temp_test_docjump.py")
