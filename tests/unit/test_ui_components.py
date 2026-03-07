import unittest
import os
import sys
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
    msg_index=msg_index
)
"""
        with open("temp_test_ui.py", "w", encoding="utf-8") as f:
            f.write(script_content)
            
        try:
            # AppTest를 사용하여 스크립트 실행
            at = AppTest.from_file("temp_test_ui.py").run()
            
            # 1. 버튼들 확인 (숫자만 포함된 라벨)
            button_labels = [b.label for b in at.button]
            
            print(f"발견된 버튼 라벨: {button_labels}")
            
            # 'p'가 포함된 숫자 라벨 확인
            self.assertIn("1p", button_labels, "1p 버튼이 없습니다.")
            self.assertIn("12p", button_labels, "12p 버튼이 없습니다.")            
            # 2. 버튼 키 접두사 확인 (12p 버튼)
            jump_button_12 = next(b for b in at.button if b.label == "12p")
            self.assertTrue(jump_button_12.key.startswith("jump_0_12_"), f"버튼 키 오류: {jump_button_12.key}")
            
            # 3. CSS 제거 확인 (기존 스타일 코드가 없는지)
            style_markdowns = [m for m in at.markdown if "white-space: nowrap !important" in m.value]
            self.assertEqual(len(style_markdowns), 0, "버튼 스타일 CSS가 여전히 존재합니다.")
            
            print("✅ UI 버튼 숫자 표기 및 CSS 제거 테스트 통과")
            
        finally:
            if os.path.exists("temp_test_ui.py"):
                os.remove("temp_test_ui.py")

if __name__ == "__main__":
    unittest.main()
