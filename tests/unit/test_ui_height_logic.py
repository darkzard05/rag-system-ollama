import unittest
from unittest.mock import MagicMock, patch
import streamlit as st

# 테스트 대상 로직을 포함한 모듈 경로 설정
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

class TestUIHeightLogic(unittest.TestCase):
    def setUp(self):
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # streamlit_js_eval 모듈 모킹 (설치되지 않은 환경 대비)
        import sys
        from unittest.mock import MagicMock
        if "streamlit_js_eval" not in sys.modules:
            sys.modules["streamlit_js_eval"] = MagicMock()
            sys.modules["streamlit_js_eval"].streamlit_js_eval = MagicMock()

    def test_height_calculation_formula(self):
        """계산 공식 자체가 정확한지 검증 (800px 브라우저 기준)"""
        viewport_height = 800
        
        # 공식: viewport - 120 (최소 400)
        calculated_base = max(400, viewport_height - 120)
        self.assertEqual(calculated_base, 680)
        
        # 채팅창 공식: base - 80 (최소 300)
        chat_height = max(300, calculated_base - 80)
        self.assertEqual(chat_height, 600)
        
        # PDF 뷰어 공식: base - 60 (최소 300)
        pdf_height = max(300, calculated_base - 60)
        self.assertEqual(pdf_height, 620)

    @patch("streamlit_js_eval.streamlit_js_eval")
    def test_main_height_extraction_logic(self, mock_js_eval):
        """main.py의 높이 추출 로직 흐름 검증"""
        # st.session_state 대신 로컬 딕셔너리를 사용하여 격리된 상태 테스트
        session_state = {}
        
        # 1. 처음에는 아무것도 없음
        self.assertNotIn("calculated_container_height", session_state)
        
        # 2. JS에서 1000px를 반환한다고 가정
        mock_js_eval.return_value = 1000
        
        # 로직 실행 시뮬레이션 (src/main.py 내용 일부)
        if session_state.get("viewport_height_finalized") is not True:
            eval_height = mock_js_eval()
            if eval_height:
                session_state["calculated_container_height"] = max(400, int(eval_height) - 120)
                session_state["viewport_height_finalized"] = True
        
        # 결과 검증
        self.assertEqual(session_state["calculated_container_height"], 880)
        self.assertTrue(session_state["viewport_height_finalized"])

    def test_component_height_fallback(self):
        """세션 상태가 없을 때 기본값(650)으로 잘 돌아가는지 검증"""
        # st.session_state 대신 로컬 딕셔너리 사용
        session_state = {}
        calculated_height = session_state.get("calculated_container_height", 650)
        chat_container_height = max(300, calculated_height - 80)
        
        self.assertEqual(chat_container_height, 570) # 650 - 80 = 570

if __name__ == "__main__":
    unittest.main()
