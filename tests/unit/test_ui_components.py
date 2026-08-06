import os
import sys
import unittest

from streamlit.testing.v1 import AppTest

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


def test_timeline_fragment_poll_interval_from_config():
    """타임라인 fragment의 run_every가 config의 UI_TIMELINE_POLL_SECONDS를 사용한다.

    decorator는 모듈 import 시점에 적용되므로, streamlit.fragment를 mock한 뒤
    chat 모듈을 reload하여 전달된 run_every를 검증한다.
    """
    import importlib
    from unittest.mock import patch

    import streamlit

    from common.config import UI_TIMELINE_POLL_SECONDS

    with patch.object(streamlit, "fragment") as mock_fragment:
        # decorator가 함수를 그대로 반환하도록 no-op 처리
        mock_fragment.return_value = lambda func: func

        import ui.components.chat as chat_module

        importlib.reload(chat_module)

        mock_fragment.assert_called_once()
        _, kwargs = mock_fragment.call_args
        assert kwargs.get("run_every") == UI_TIMELINE_POLL_SECONDS

    # 원래 fragment 래핑 상태로 복원 (다른 테스트에 영향 방지)
    importlib.reload(chat_module)
