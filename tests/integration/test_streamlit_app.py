import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestRAGStreamlitUI(unittest.TestCase):
    def setUp(self):
        """앱 테스트 인스턴스 초기화 (src/main.py가 진입점이라고 가정)"""
        self.at = AppTest.from_file("src/main.py", default_timeout=30)

    def test_app_initial_state(self):
        """앱 시작 시 초기 UI 요소들이 존재하는지 검증"""
        self.at.run()

        # 사이드바 제목 확인
        self.assertTrue(self.at.sidebar.header[0].value.strip() != "")
        # 채팅 환영 메시지 존재 여부
        self.assertTrue(any("환영합니다" in str(info.value) for info in self.at.info))
        # 모델 선택 셀렉트박스 존재 여부
        self.assertTrue(len(self.at.sidebar.selectbox) >= 2)

    def test_chat_interaction_rendering(self):
        """채팅 입력 시 화면 렌더링 흐름 검증"""
        self.at.run()

        # 1. 채팅 입력 시뮬레이션
        if hasattr(self.at, "chat_input") and self.at.chat_input:
            prompt = self.at.chat_input[0]
            prompt.set_value("테스트 질문입니다.").run()

            # 2. 사용자 메시지가 화면에 렌더링되었는지 확인
            user_msg = [m for m in self.at.chat_message if m.name == "user"]
            self.assertGreater(len(user_msg), 0)

            # 3. 답변 생성 중 상태 박스(Status Box) 렌더링 확인
            # (로그가 HTML 형태로 출력되므로 markdown 요소 중 status-container를 포함하는지 확인)
            status_markdowns = [
                m for m in self.at.markdown if "status-container" in str(m.value)
            ]
            self.assertGreater(len(status_markdowns), 0)

            print("✅ UI 상호작용 및 기본 렌더링 테스트 통과")


if __name__ == "__main__":
    # 실제 환경에서는 Streamlit 앱이 복잡하므로 일부 기능만 단위 테스트로 실행
    unittest.main()
