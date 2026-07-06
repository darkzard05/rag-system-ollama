import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# 프로젝트 루트 및 src 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.ui.components.chat import render_message


class TestChatMetricsRendering(unittest.TestCase):
    """채팅창 성능 지표 UI 렌더링 검증 테스트"""

    def setUp(self):
        # 개별 함수 패치
        self.patch_markdown = patch("ui.components.chat.st.markdown")
        self.patch_popover = patch("ui.components.chat.st.popover")
        self.patch_chat_message = patch("ui.components.chat.st.chat_message")
        self.patch_container = patch("ui.components.chat.st.container")
        
        self.mock_markdown = self.patch_markdown.start()
        self.mock_popover = self.patch_popover.start()
        self.mock_chat_message = self.patch_chat_message.start()
        self.mock_container = self.patch_container.start()

    def tearDown(self):
        patch.stopall()

    def get_rendered_html(self, role="assistant", metrics=None):
        """render_message를 호출하여 st.markdown에 전달된 HTML 중 성능 테이블을 추출합니다."""
        # popover context manager mock 처리:
        popover_container = MagicMock()
        self.mock_popover.return_value = popover_container
        popover_container.__enter__.return_value = popover_container
        
        # container context manager mock 처리:
        container_container = MagicMock()
        self.mock_container.return_value = container_container
        container_container.__enter__.return_value = container_container

        render_message(
            role=role,
            content="테스트 답변입니다.",
            metrics=metrics,
            wrap_in_container=False,
        )

        # st.markdown 호출 중 <table class="perf-table">가 포함된 인자를 찾음
        for call in self.mock_markdown.call_args_list:
            args, kwargs = call
            content = args[0] if args else kwargs.get("content", "")
            if isinstance(content, str) and '<div class="perf-grid">' in content:
                return content
        return None

    def test_basic_metrics_rendering(self):
        """기본 성능 지표가 테이블에 정확한 수치로 출력되는지 검증합니다."""
        metrics = {
            "total_time": 2.345,
            "tps": 32.12,
            "input_token_count": 150,
            "token_count": 250,
        }

        html = self.get_rendered_html(metrics=metrics)

        self.assertIsNotNone(html, "성능 지표 테이블이 렌더링되지 않았습니다.")
        self.assertIn("2.3s", html)
        self.assertIn("32.1 t/s", html)
        self.assertIn("150 / 250", html)
        self.assertIn("400 total", html)

    def test_excellent_status_rendering(self):
        """최상급(Excellent) 상태의 레이블과 클래스가 적용되는지 검증합니다."""
        metrics = {
            "total_time": 0.5,  # < 1.5: Excellent
            "tps": 50.0,  # > 40: Fast
            "input_token_count": 10,
            "token_count": 20,
        }

        html = self.get_rendered_html(metrics=metrics)

        self.assertIn("✨ Excellent", html)
        self.assertIn("🚀 Fast", html)

    def test_poor_status_rendering(self):
        """최악급(Poor) 상태의 레이블과 클래스가 적용되는지 검증합니다."""
        metrics = {
            "total_time": 5.0,  # >= 3.5: Poor
            "tps": 10.0,  # <= 20: Slow
            "input_token_count": 10,
            "token_count": 20,
        }

        html = self.get_rendered_html(metrics=metrics)

        self.assertIn("⚠️ Poor", html)
        self.assertIn("🐢 Slow", html)

    def test_stable_status_rendering(self):
        """보통(Stable) 상태의 레이블과 클래스가 적용되는지 검증합니다."""
        metrics = {
            "total_time": 2.0,  # 1.5 <= x < 3.5: Stable
            "tps": 30.0,  # 20 < x <= 40: Normal
            "input_token_count": 10,
            "token_count": 20,
        }

        html = self.get_rendered_html(metrics=metrics)

        self.assertIn("🟢 Stable", html)
        self.assertIn("🟢 Normal", html)

    def test_no_metrics_no_rendering(self):
        """metrics가 없을 때 성능 지표 테이블이 출력되지 않는지 검증합니다."""
        html = self.get_rendered_html(metrics=None)
        self.assertIsNone(
            html, "metrics가 없음에도 성능 지표 테이블이 렌더링되었습니다."
        )

    def test_user_role_no_metrics(self):
        """사용자(user) 역할일 때는 metrics가 있어도 출력되지 않는지 검증합니다."""
        metrics = {"total_time": 1.0, "tps": 10.0}
        html = self.get_rendered_html(role="user", metrics=metrics)
        self.assertIsNone(html, "사용자 메시지에는 성능 지표가 출력되어서는 안 됩니다.")


if __name__ == "__main__":
    unittest.main()
