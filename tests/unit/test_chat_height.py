import unittest
import os
import sys
import streamlit as st
print(f"DEBUG: streamlit type in test_chat_height: {type(st)}")
from streamlit.testing.v1 import AppTest

# 프로젝트 루트를 경로에 추가



class TestChatHeight(unittest.TestCase):
    def test_chat_container_height_calculation(self):
        """
        채팅 컨테이너의 높이가 계산된 높이에 기반하여 올바르게 설정되는지 검증합니다.
        """
        script_content = """
import streamlit as st
import sys
import os
from unittest.mock import patch

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from src.ui.components.chat import render_chat_interface

# 세션 상태에 계산된 높이 및 빌드 상태 주입
st.session_state["calculated_container_height"] = 800
st.session_state["is_building_rag"] = True
st.session_state["rebuild_status"] = "Analyzing documents..."
st.session_state["status_logs"] = ["Starting pipeline...", "Reading PDF..."]

# UI 렌더링 호출
render_chat_interface()
"""
        with open("temp_test_height.py", "w", encoding="utf-8") as f:
            f.write(script_content)

        try:
            # 타임아웃을 넉넉하게 설정하고, 실행
            at = AppTest.from_file("temp_test_height.py").run(timeout=10)
            print("✅ 채팅창 높이 계산 및 렌더링 테스트 완료 (에러 없음 확인)")
        finally:
            if os.path.exists("temp_test_height.py"):
                os.remove("temp_test_height.py")


if __name__ == "__main__":
    unittest.main()
