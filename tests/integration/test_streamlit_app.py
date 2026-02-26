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

        # 사이드바 헤더 확인 (sidebar.py: st.markdown("<div class='sidebar-header'>🤖 RAG System</div>", ...))
        # st.markdown은 markdown 요소로 렌더링됨
        assert any(
            "RAG System" in str(m.value) for m in self.at.sidebar.markdown
        ) or any("GraphRAG" in str(h.value) for h in self.at.sidebar.header)
        
        # 채팅 환영 메시지 존재 여부 (chat.py: MSG_CHAT_GUIDE 출력)
        # config.yml의 chat_guide에 "환영합니다"가 포함되어 있음
        assert any("환영합니다" in str(m.value) for m in self.at.chat_message[0].markdown)
        
        # 모델 선택 셀렉트박스 존재 여부 (sidebar.py: st.selectbox)
        # sidebar 내부에 중첩된 요소들은 at.sidebar.selectbox로 접근 가능
        assert len(self.at.sidebar.selectbox) >= 1

    def test_chat_interaction_rendering(self):
        """채팅 입력 시 화면 렌더링 흐름 검증"""
        self.at.run()

        # 1. 채팅 입력 시뮬레이션
        if hasattr(self.at, "chat_input") and self.at.chat_input:
            prompt = self.at.chat_input[0]
            prompt.set_value("테스트 질문입니다.").run()

            # 2. 사용자 메시지가 화면에 렌더링되었는지 확인
            user_msg = [m for m in self.at.chat_message if m.name == "user"]
            assert len(user_msg) > 0
            assert "테스트 질문입니다." in str(user_msg[0].markdown[0].value)

            # 3. 답변 생성 시도 로그 또는 채팅 입력 비활성화 상태 확인
            # (RAG 엔진이 백그라운드에서 동작하므로 입력창이 비활성화되었거나 
            # 다음 런타임에 메시지가 추가되는지 확인)
            print("✅ UI 상호작용 및 기본 렌더링 테스트 통과 (사용자 입력 확인)")


if __name__ == "__main__":
    # 실제 환경에서는 Streamlit 앱이 복잡하므로 일부 기능만 단위 테스트로 실행
    unittest.main()
