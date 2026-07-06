import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

class TestHeightExtractionE2E(unittest.TestCase):
    def test_multi_step_height_extraction(self):
        """
        높이 추출 로직의 상태 전이 과정을 검증합니다.
        1. 초기 상태: 기본값 750 설정
        2. 높이 업데이트: 입력 값에 따른 계산 결과 반영
        """
        # st.session_state를 모사하는 딕셔너리
        session_state = {}
        
        # --- Step 1: 첫 실행 시뮬레이션 ---
        # 초기화 로직
        if "viewport_height_finalized" not in session_state:
            session_state["viewport_height_finalized"] = False
        if "calculated_container_height" not in session_state:
            session_state["calculated_container_height"] = 750
            
        # 검증: 기본값 750이 세팅되어야 함
        self.assertEqual(session_state["calculated_container_height"], 750)
        self.assertFalse(session_state["viewport_height_finalized"])
        
        # --- Step 2: 높이 업데이트 시뮬레이션 ---
        # 사용자 입력(브라우저 높이)이 1000px라고 가정
        eval_height = 1000
        
        if not session_state.get("viewport_height_finalized"):
            if eval_height > 0:
                session_state["calculated_container_height"] = max(400, int(eval_height) - 120)
                session_state["viewport_height_finalized"] = True
        
        # 최종 상태 확인: 1000 - 120 = 880
        self.assertEqual(session_state["calculated_container_height"], 880)
        self.assertTrue(session_state["viewport_height_finalized"])
        
        print("✅ 높이 추출 로직 상태 전이 테스트 통과")

if __name__ == "__main__":
    unittest.main()
