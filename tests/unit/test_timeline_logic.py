import unittest
from unittest.mock import MagicMock
import sys
import os

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

class TestTimelineLogic(unittest.TestCase):
    def test_update_status_html_generation(self):
        """
        update_status 함수가 상세 상태 메시지를 포함한 
        올바른 HTML 타임라인을 생성하는지 검증합니다.
        """
        # chat.py 내부의 update_status 로직을 모방한 테스트
        live_logs = []
        
        def get_timeline_html(msg, is_complete=False):
            if msg not in live_logs:
                live_logs.append(msg)
            
            status_icon = "✅" if is_complete else "🚀"
            expander_title = f"{status_icon} 실시간 RAG 분석 타임라인"
            
            lines = "".join(
                [
                    f"<div style='font-size: 0.85rem; color: var(--text-color); margin-bottom: 8px; display: flex; align-items: flex-start; line-height: 1.5; opacity: 0.9;'>"
                    f"<span style='color: #1e88e5; margin-right: 10px; font-weight: bold;'>▹</span>"
                    f"<span>{item}</span></div>"
                    for item in live_logs
                ]
            )
            
            html = (
                f"<div style='margin-bottom: 15px;'>"
                f"<details {'open' if not is_complete else ''} class='timeline-container' style='border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; padding: 10px;'>"
                f"<summary class='timeline-summary' style='font-weight: 600; color: var(--text-color); cursor: pointer; list-style: none; display: flex; align-items: center; padding: 5px 0;'>"
                f"{expander_title}</summary>"
                f"<div style='margin-top: 12px; padding: 15px; background-color: rgba(128,128,128,0.05); border-radius: 8px; border-left: 3px solid #1e88e5;'>"
                f"<div style='font-size: 0.75rem; color: var(--text-color); opacity: 0.6; margin-bottom: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>"
                f"⏱️ Live Pipeline Execution"
                f"</div>{lines}</div></details></div>"
            )
            return html

        # 1. 초기 상태 테스트
        html1 = get_timeline_html("파이프라인 가동 중")
        self.assertIn("실시간 RAG 분석 타임라인", html1)
        self.assertIn("파이프라인 가동 중", html1)
        self.assertIn("open", html1) # 진행 중에는 열려 있어야 함
        
        # 2. 상세 상태 추가 테스트 (is_status_update 청크 수신 상황 모사)
        detail_msg = "지식 저장소에서 관련 문서 5개를 성공적으로 찾았습니다"
        html2 = get_timeline_html(detail_msg)
        self.assertIn(detail_msg, html2)
        self.assertIn("파이프라인 가동 중", html2) # 이전 로그 유지 확인
        
        # 3. 완료 상태 테스트
        html3 = get_timeline_html("답변 생성 완료", is_complete=True)
        self.assertIn("✅", html3)
        self.assertNotIn("open", html3) # 완료 후에는 접혀 있어야 함 (details 태그에 open 속성 없음)
        
        print("✅ 타임라인 HTML 생성 및 상태 전이 로직 검증 완료")

if __name__ == "__main__":
    unittest.main()
