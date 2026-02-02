import os
import sys
import threading
import time
import unittest
from pathlib import Path

# --- 경로 설정 최적화 (절대 경로 기반) ---
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.thread_safe_session import ThreadSafeSessionManager


class TestPerSessionLock(unittest.TestCase):
    def test_per_session_isolation(self):
        manager = ThreadSafeSessionManager()

        def user_task(session_id, sleep_time):
            manager.set_session_id(session_id)
            manager.init_session()

            with manager._acquire_lock():
                # Lock을 잡은 상태에서 대기
                # 만약 global lock이라면 다른 유저가 이 동안 block되어야 함
                time.sleep(sleep_time)
                manager.set("done", True)

        start_time = time.time()

        # User 1이 lock을 0.5초간 잡음
        t1 = threading.Thread(target=user_task, args=("user_1", 0.5))
        # User 2는 lock을 즉시 잡아야 함 (별도 세션이므로)
        t2 = threading.Thread(target=user_task, args=("user_2", 0.1))

        t1.start()
        time.sleep(0.05)  # t1이 lock을 확실히 잡을 때까지 잠시 대기
        t2.start()

        t1.join()
        t2.join()

        end_time = time.time()

        # 만약 전역 락이었다면 최소 0.5 + 0.1 = 0.6초 이상 걸려야 함
        # 세션별 락이라면 t2는 t1을 기다리지 않으므로 약 0.5초 근처에서 끝나야 함
        duration = end_time - start_time
        print(f"Total duration: {duration:.4f}s")

        # 0.6초보다 적게 걸렸다면 병렬 처리가 된 것임
        assert duration < 0.6


if __name__ == "__main__":
    unittest.main()
