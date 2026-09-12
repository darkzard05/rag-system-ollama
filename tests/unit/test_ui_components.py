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
        '근거 페이지로 이동' 버튼이 숫자만 표시하도록 정상적으로 생성되는지 검증합니다.
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

            # 1. 버튼들 확인 (숫자만 포함된 라벨)
            button_labels = [b.label for b in at.button]

            # 'p'가 포함된 숫자 라벨 확인
            assert "1p" in button_labels, "1p 버튼이 없습니다."
            assert "12p" in button_labels, "12p 버튼이 없습니다."
            # 2. 버튼 키 접두사 확인 (12p 버튼)
            jump_button_12 = next(b for b in at.button if b.label == "12p")
            assert jump_button_12.key.startswith("jump_msg_0_12_"), (
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
    핸들러를 정확히 1회 호출한다. 렌더 단계 인라인 호출로 회귀하면
    StreamlitAPIException(위젯키 스크립트 대입)이 재발한다."""
    script_content = """
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from unittest.mock import MagicMock
from ui.components.chat_references import _render_references_content

doc = MagicMock()
doc.metadata = {"page": 3}
if "__jumped_pages" not in st.session_state:
    st.session_state["__jumped_pages"] = []

def on_jump(p):
    st.session_state["__jumped_pages"].append(p)

_render_references_content(
    msg_id="m1",
    documents=[doc],
    on_page_jump=on_jump,
    citations=None,
    generating=False,
)
"""
    with open("temp_test_jump_callback.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    try:
        at = AppTest.from_file("temp_test_jump_callback.py").run(timeout=60)
        jump_btn = next(b for b in at.button if b.label == "3p")
        at_after = jump_btn.click().run(timeout=60)
        assert at_after.session_state["__jumped_pages"] == [3]
    finally:
        if os.path.exists("temp_test_jump_callback.py"):
            os.remove("temp_test_jump_callback.py")


def test_doc_jump_click_invokes_handler_via_callback():
    """P0 회귀: doc 점프 버튼 클릭이 on_click 콜백으로 세션에
    pdf_target_page 토큰(page=문서 메타)을 세팅한다."""
    script_content = """
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from unittest.mock import MagicMock
from ui.components.chat_references import _render_references_content
from core.session import SessionManager

SessionManager.set_session_id("t_docjump")

doc = MagicMock()
doc.metadata = {"page": 7, "doc_id": "docA"}
SessionManager.set("documents", [doc], "t_docjump")

_render_references_content(
    msg_id="m1",
    documents=[doc],
    on_page_jump=None,
    citations=[{"doc_id": "docA", "section": "Intro"}],
    generating=False,
)
"""
    SessionManager.reset_all_state("t_docjump")
    with open("temp_test_docjump.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    try:
        at = AppTest.from_file("temp_test_docjump.py").run(timeout=60)
        btn = next(b for b in at.button if (b.label or "").startswith("1. "))
        at_after = btn.click().run(timeout=60)
        # P0 계약: 클릭이 on_click 콜백 페이즈에서 navigate_to_page를 실행한다.
        # (수정 전: 렌더 단계 인라인 호출 → StreamlitAPIException으로 이미
        # 인스턴스화된 pdf_nav_input_v6 위젯 키 대입에 실패). session_state의
        # 내비게이션 부작용이 버전 무관 검증 지점이며, AppTest는 매 rerun 시작에
        # 자체 default sid로 리셋하므로 SessionManager 토큰 위치는 단언하지 않는다.
        assert not at_after.exception, at_after.exception
        assert at_after.session_state["pdf_nav_input_v6"] == 1
    finally:
        if os.path.exists("temp_test_docjump.py"):
            os.remove("temp_test_docjump.py")
