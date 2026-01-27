import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.thread_safe_session import ThreadSafeSessionManager
from core.session import SessionManager

class StreamlitUILifecycleSimulator(unittest.TestCase):
    def setUp(self):
        """실제 브라우저와 같은 환경 설정"""
        self.session_id = "ui_simulation_session_001"
        self.shared_state = {} # 실제 st.session_state 역할을 할 저장소
        
        # 초기화
        ThreadSafeSessionManager.clear_session(self.session_id)
        ThreadSafeSessionManager.set_session_id(self.session_id)

    def mock_app_render(self, user_input_trigger=None):
        """
        src/ui/ui.py의 render_chat_interface 로직을 시뮬레이션합니다.
        Streamlit의 렌더링 순서(기존 메시지 출력 -> 입력창 표시)를 엄격히 따릅니다.
        """
        # 1. Streamlit 환경 패치
        with patch('core.thread_safe_session.st.session_state', self.shared_state), \
             patch('core.thread_safe_session.get_script_run_ctx') as mock_ctx_mgr, \
             patch('streamlit.chat_input') as mock_chat_input, \
             patch('streamlit.chat_message') as mock_chat_msg:
            
            mock_ctx = MagicMock()
            mock_ctx.session_id = self.session_id
            mock_ctx_mgr.return_value = mock_ctx
            
            # --- [APP START] ---
            SessionManager.init_session()
            
            # [UI 컴포넌트 1] 기존 메시지 히스토리 렌더링
            messages = SessionManager.get_messages()
            print(f"\n[UI] {len(messages)}개의 기존 메시지를 화면에 그립니다.")
            for msg in messages:
                # 실제로 렌더링 함수가 호출되는지 확인
                mock_chat_msg(msg["role"])
            
            # [UI 컴포넌트 2] 채팅 입력 위젯 (이전 입력값이 있으면 실행됨)
            if user_input_trigger:
                print(f"[Input Widget] 사용자가 '{user_input_trigger}'를 입력했습니다.")
                
                # 메시지 추가
                SessionManager.add_message("user", user_input_trigger)
                
                # 비동기 답변 생성 프로세스 (Sync 로직 포함)
                SessionManager.sync_to_fallback()
                
                async def generate_response_flow():
                    # 비동기 스레드 모사 (Streamlit 접근 불가)
                    with patch('core.thread_safe_session.ThreadSafeSessionManager._is_streamlit_available', return_value=False):
                        ans = f"'{user_input_trigger}'에 대한 전문 보고서 답변입니다. [p.1]"
                        SessionManager.add_message("assistant", ans)
                
                loop = asyncio.get_event_loop()
                loop.run_until_complete(generate_response_flow())
                
                SessionManager.sync_from_fallback()
                print("[Sync] 비동기 답변을 메인 세션으로 가져왔습니다.")
            
            # --- [APP END] ---

    def test_ui_history_persistence_on_consecutive_inputs(self):
        """연속 입력 시 UI에 이전 답변이 남는지 테스트"""
        
        print("\n--- [Run 1] 사용자의 첫 번째 질문 ---")
        self.mock_app_render(user_input_trigger="RAG가 무엇인가요?")
        
        # 첫 번째 질문 후 상태 확인 (User 1, Assistant 1)
        self.assertEqual(len(self.shared_state["messages"])
, 2)
        
        print("\n--- [Run 2] 리런(Rerun) 후 UI 렌더링 ---")
        # 입력 없이 페이지만 다시 그려지는 상황
        self.mock_app_render()
        
        # 여전히 데이터가 있어야 함
        self.assertEqual(len(self.shared_state["messages"])
, 2)
        
        print("\n--- [Run 3] 사용자의 두 번째 질문 ---")
        self.mock_app_render(user_input_trigger="Ollama 설정 방법을 알려주세요.")
        
        # 최종 확인
        final_history = self.shared_state["messages"]
        print(f"\n[최종 UI 히스토리 데이터]")
        for i, m in enumerate(final_history):
            print(f"  {i+1}. {m['role'].upper()}: {m['content'][:30]}...")
            
        self.assertEqual(len(final_history), 4, "이전 답변이 사라지지 않고 총 4개의 메시지가 있어야 합니다.")
        self.assertEqual(final_history[1]["role"], "assistant", "첫 번째 답변이 유지되어야 합니다.")
        self.assertEqual(final_history[3]["role"], "assistant", "두 번째 답변이 생성되어야 합니다.")
        
        print("\n✅ [성공] Streamlit UI 라이프사이클 시뮬레이션 결과, 데이터 유실 없이 답변이 누적됩니다.")

if __name__ == "__main__":
    unittest.main()
