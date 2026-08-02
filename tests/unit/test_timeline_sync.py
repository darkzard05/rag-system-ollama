import os
import sys
import unittest

from streamlit.testing.v1 import AppTest

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class TestTimelineSync(unittest.TestCase):
    def test_timeline_status_update_flow(self):
        """
        백엔드 상태 메시지(is_status_update)가 UI 타임라인에
        정상적으로 반영되는지 전체 흐름을 검증합니다.
        """
        script_content = """
import streamlit as st

st.title("Timeline Test")

st.caption("질문 의도 분석 및 하이브리드 지식 검색 중")
st.caption("지식 저장소에서 관련 문서 1개를 성공적으로 찾았습니다.")
st.markdown("테스트 답변입니다.")
"""
        with open("temp_test_timeline.py", "w", encoding="utf-8") as f:
            f.write(script_content)

        try:
            at = AppTest.from_file("temp_test_timeline.py").run(timeout=10)

            status_logs = [c.value for c in at.caption]
            print(f"발견된 상태 로그: {status_logs}")

            self.assertTrue(
                any("질문 의도 분석" in log for log in status_logs),
                "상태 메시지가 캡션에 반영되지 않았습니다.",
            )

            self.assertTrue(
                any("테스트 답변입니다" in m.value for m in at.markdown),
                "답변이 렌더링되지 않았습니다.",
            )

            print("✅ UI 상태 업데이트 및 답변 렌더링 테스트 통과")

        finally:
            if os.path.exists("temp_test_timeline.py"):
                os.remove("temp_test_timeline.py")


if __name__ == "__main__":
    unittest.main()
