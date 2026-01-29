import unittest
import sys
from pathlib import Path

# 프로젝트 루트 및 src 디렉토리 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from streamlit.testing.v1 import AppTest


class TestChatBoxRendering(unittest.TestCase):
    def setUp(self):
        # main.py 경로 설정
        self.app_path = str(ROOT_DIR / "src" / "main.py")
        self.at = AppTest.from_file(self.app_path, default_timeout=30)

    def test_assistant_box_html_content(self):
        """채팅 박스 내부에 HTML 툴팁이 포함된 마크다운이 정확히 렌더링되는지 테스트"""

        # 1. 앱의 session_state에 직접 데이터 주입
        # 실제 앱의 SessionManager가 사용하는 키 구조를 따름
        sample_answer = 'The capital is Paris <span class="tooltip">[p.1]<span class="tooltip-text">Source content...</span></span>.'

        # Streamlit AppTest에서 session_state 초기화
        self.at.run()  # 먼저 한 번 실행하여 상태 생성

        # 메시지 리스트 강제 주입
        self.at.session_state["messages"] = [
            {"role": "user", "content": "Where is the capital?"},
            {"role": "assistant", "content": sample_answer},
        ]

        # 2. 상태 변경 후 다시 실행 (Rerun)
        self.at.run()

        # 3. 채팅 박스 내부 조사
        # at.chat_message를 통해 assistant 메시지 박스 추출
        assistant_messages = [m for m in self.at.chat_message if m.name == "assistant"]

        print("\n🔍 [채팅 박스 렌더링 정밀 검사]")
        print(f"찾은 어시스턴트 박스 수: {len(assistant_messages)}")

        if len(assistant_messages) > 0:
            # 박스 안의 마크다운 텍스트 확인
            # assistant_messages[0]은 ChatMessageProxy 객체
            # 그 안의 첫 번째 마크다운 요소를 가져옴
            content = assistant_messages[0].markdown[0].value
            print(f"박스 내부 컨텐츠: {content}")

            self.assertIn(
                'class="tooltip"', content, "HTML 툴팁 태그가 소실되었습니다."
            )
            self.assertIn("[p.1]", content, "인용구 텍스트가 소실되었습니다.")
            print("✅ 채팅 박스 내부 HTML 렌더링 성공 확인!")
        else:
            # 만약 chat_message 프록시로 잡히지 않는 경우, 전체 마크다운에서 확인
            all_markdown = "".join([m.value for m in self.at.markdown])
            self.assertIn(
                sample_answer,
                all_markdown,
                "메시지가 마크다운 요소로 렌더링되지 않았습니다.",
            )
            print("✅ 전체 마크다운 렌더링 결과 내 메시지 포함 확인!")


if __name__ == "__main__":
    unittest.main()
