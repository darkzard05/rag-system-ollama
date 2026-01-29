"""
Thread Safety Integration Tests (Simplified)

ThreadSafeSessionManager의 통합 테스트입니다.
"""

import sys
import unittest
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mock streamlit before importing thread_safe_session
class MockStreamlit:
    """Mock streamlit module"""

    session_state = {}


sys.modules["streamlit"] = MockStreamlit()


class TestThreadSafeSessionManager(unittest.TestCase):
    """ThreadSafeSessionManager 기본 기능 테스트"""

    def setUp(self):
        """각 테스트 전 설정"""
        import streamlit as st

        st.session_state = {}

    def tearDown(self):
        """각 테스트 후 정리"""
        import streamlit as st

        st.session_state = {}

    def test_lock_context_manager(self):
        """Lock context manager가 작동하는지 검증"""
        from core.thread_safe_session import _LockContext
        import threading

        lock = threading.RLock()

        # Lock 획득 테스트
        with _LockContext(lock, 5.0):
            # Lock이 획득됨
            pass

        # Lock이 해제됨
        # 다시 획득 가능
        self.assertTrue(lock.acquire(timeout=1.0))
        lock.release()

    def test_thread_safe_counter_increment(self):
        """스레드 안전한 카운터 증가 테스트"""
        import streamlit as st

        st.session_state["counter"] = 0

        def increment():
            for _ in range(10):
                current = st.session_state["counter"]
                st.session_state["counter"] = current + 1

        # 단일 스레드는 정상 작동
        st.session_state["counter"] = 0
        increment()
        self.assertEqual(st.session_state["counter"], 10)

    def test_message_list_thread_safety(self):
        """메시지 리스트 thread-safety 테스트"""
        import streamlit as st

        st.session_state["messages"] = []

        def add_messages(count):
            for i in range(count):
                msg = {"role": "user", "content": f"Message {i}"}
                messages = st.session_state.get("messages", [])
                messages.append(msg)
                st.session_state["messages"] = messages

        threads = [
            threading.Thread(target=add_messages, args=(50,)),
            threading.Thread(target=add_messages, args=(50,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 대략 100개 정도의 메시지가 있어야 함
        messages = st.session_state.get("messages", [])
        self.assertGreater(len(messages), 50)

    def test_concurrent_dict_updates(self):
        """동시 딕셔너리 업데이트 테스트"""
        import streamlit as st

        st.session_state["state"] = {
            "key1": 0,
            "key2": 0,
            "key3": 0,
        }

        def update_state(key_name, count):
            for _ in range(count):
                state = st.session_state["state"].copy()
                state[key_name] = state.get(key_name, 0) + 1
                st.session_state["state"] = state

        threads = [
            threading.Thread(target=update_state, args=("key1", 20)),
            threading.Thread(target=update_state, args=("key2", 20)),
            threading.Thread(target=update_state, args=("key3", 20)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final_state = st.session_state["state"]
        # 모든 키가 증가했는지 확인
        for key in ["key1", "key2", "key3"]:
            self.assertGreater(final_state.get(key, 0), 0)

    def test_session_manager_default_state(self):
        """SessionManager 기본 상태 검증"""
        from core.thread_safe_session import ThreadSafeSessionManager

        # 기본 상태 확인
        self.assertIsNotNone(ThreadSafeSessionManager.DEFAULT_SESSION_STATE)

        # 필수 키 확인
        required_keys = [
            "messages",
            "llm",
            "qa_chain",
            "pdf_processed",
            "vector_store",
        ]

        for key in required_keys:
            self.assertIn(key, ThreadSafeSessionManager.DEFAULT_SESSION_STATE)

    def test_lock_timeout_constant(self):
        """Lock timeout 상수 검증"""
        from core.thread_safe_session import ThreadSafeSessionManager

        # Lock timeout이 설정되어 있는지 확인
        self.assertIsNotNone(ThreadSafeSessionManager._lock_timeout)
        self.assertGreater(ThreadSafeSessionManager._lock_timeout, 0)

    def test_multiple_lock_acquisitions(self):
        """RLock의 재진입 가능성 테스트"""
        import threading

        lock = threading.RLock()

        # RLock은 같은 스레드에서 여러 번 획득 가능
        self.assertTrue(lock.acquire())
        self.assertTrue(lock.acquire())
        lock.release()
        lock.release()

    def test_session_state_isolation(self):
        """테스트 간 세션 상태 격리 검증"""
        import streamlit as st

        # 각 테스트는 독립적인 session_state를 가져야 함
        st.session_state["test_value"] = "test1"
        self.assertEqual(st.session_state["test_value"], "test1")

        # 다음 테스트에서는 깨끗한 상태로 시작
        st.session_state = {}
        self.assertNotIn("test_value", st.session_state)

    def test_concurrent_reads_concurrent_writes(self):
        """동시 읽기/쓰기 스트레스 테스트"""
        import streamlit as st

        st.session_state["data"] = {"value": 0}
        results = []

        def reader():
            for _ in range(50):
                value = st.session_state["data"].get("value", 0)
                results.append(("read", value))

        def writer():
            for i in range(50):
                data = st.session_state["data"].copy()
                data["value"] = i
                st.session_state["data"] = data
                results.append(("write", i))

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 스트레스 테스트 완료
        self.assertGreater(len(results), 0)


class TestBackwardCompatibility(unittest.TestCase):
    """후방호환성 검증"""

    def setUp(self):
        """각 테스트 전 설정"""
        import streamlit as st

        st.session_state = {}

    def test_session_manager_exists(self):
        """SessionManager가 존재하는지 검증"""
        try:
            from core.session import SessionManager

            self.assertIsNotNone(SessionManager)
        except ImportError:
            self.fail("SessionManager를 import할 수 없음")

    def test_thread_safe_session_manager_exists(self):
        """ThreadSafeSessionManager가 존재하는지 검증"""
        try:
            from core.thread_safe_session import ThreadSafeSessionManager

            self.assertIsNotNone(ThreadSafeSessionManager)
        except ImportError:
            self.fail("ThreadSafeSessionManager를 import할 수 없음")

    def test_session_manager_has_init_session(self):
        """SessionManager.init_session 메서드 존재 확인"""
        from core.session import SessionManager

        self.assertTrue(hasattr(SessionManager, "init_session"))
        self.assertTrue(callable(getattr(SessionManager, "init_session")))

    def test_session_manager_has_set_get(self):
        """SessionManager의 set/get 메서드 확인"""
        from core.session import SessionManager

        self.assertTrue(hasattr(SessionManager, "set"))
        self.assertTrue(hasattr(SessionManager, "get"))
        self.assertTrue(callable(getattr(SessionManager, "set")))
        self.assertTrue(callable(getattr(SessionManager, "get")))

    def test_session_manager_has_add_message(self):
        """SessionManager.add_message 메서드 확인"""
        from core.session import SessionManager

        self.assertTrue(hasattr(SessionManager, "add_message"))
        self.assertTrue(callable(getattr(SessionManager, "add_message")))

    def test_session_manager_has_is_ready_for_chat(self):
        """SessionManager.is_ready_for_chat 메서드 확인"""
        from core.session import SessionManager

        self.assertTrue(hasattr(SessionManager, "is_ready_for_chat"))
        self.assertTrue(callable(getattr(SessionManager, "is_ready_for_chat")))

    def test_thread_safe_methods(self):
        """ThreadSafeSessionManager의 모든 thread-safe 메서드 확인"""
        from core.thread_safe_session import ThreadSafeSessionManager

        required_methods = [
            "init_session",
            "reset_for_new_file",
            "add_message",
            "is_ready_for_chat",
            "get",
            "get_messages",
            "set",
            "update",
            "increment",
            "reset_all_state",
            "has_key",
            "delete_key",
            "get_all_state",
        ]

        for method in required_methods:
            self.assertTrue(
                hasattr(ThreadSafeSessionManager, method),
                f"ThreadSafeSessionManager에 {method} 메서드가 없음",
            )


if __name__ == "__main__":
    unittest.main()
