import unittest
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.thread_safe_session import ThreadSafeSessionManager, MAX_MESSAGE_HISTORY
from core.session import SessionManager

class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        """테스트 전 세션 초기화"""
        ThreadSafeSessionManager.set_session_id("test_unit_session")
        ThreadSafeSessionManager.clear_all()
        ThreadSafeSessionManager.init_session()

    def test_add_message_persistence(self):
        """메시지가 정상적으로 누적되고 유지되는지 확인"""
        SessionManager.add_message("user", "안녕하세요")
        SessionManager.add_message("assistant", "반갑습니다!")
        
        messages = SessionManager.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

    def test_message_list_reassignment(self):
        """메시지 추가 시 리스트 객체가 재할당되는지 확인 (Streamlit 상태 감지용)"""
        storage = ThreadSafeSessionManager._get_storage()
        
        # 첫 번째 추가
        SessionManager.add_message("user", "첫 번째")
        list_v1 = storage.get("messages")
        
        # 두 번째 추가
        SessionManager.add_message("user", "두 번째")
        list_v2 = storage.get("messages")
        
        # 객체 자체가 달라야 함 (재할당 보장)
        self.assertIsNot(list_v1, list_v2)
        self.assertEqual(len(list_v2), 2)

    def test_max_history_limit(self):
        """히스토리 제한 확인"""
        for i in range(MAX_MESSAGE_HISTORY + 5):
            SessionManager.add_message("user", f"msg {i}")
        
        messages = SessionManager.get_messages()
        self.assertEqual(len(messages), MAX_MESSAGE_HISTORY)
        self.assertEqual(messages[-1]["content"], f"msg {MAX_MESSAGE_HISTORY + 4}")

if __name__ == "__main__":
    unittest.main()
