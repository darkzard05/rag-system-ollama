import sys
import threading
import time
import unittest
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import pytest

from common.exceptions import SessionLockTimeoutError
from core.thread_safe_session import ThreadSafeSessionManager


class TestSessionDeadlockPrevention(unittest.TestCase):
    def test_lock_timeout_exception(self):
        """락 점유 시간이 길 때 예외가 발생하는지 테스트"""

        manager = ThreadSafeSessionManager(lock_timeout=1.0)

        def hold_lock():
            # 1. 첫 번째 스레드에서 락을 획득하고 3초간 대기
            with manager._acquire_lock(manager):
                print("[Thread 1] 락 획득 성공, 3초간 점유합니다.")
                time.sleep(3.0)
                print("[Thread 1] 락 해제 준비.")

        t1 = threading.Thread(target=hold_lock)
        t1.start()

        # t1이 락을 확실히 잡을 때까지 잠시 대기
        time.sleep(0.5)

        # 2. 메인 스레드에서 락 획득 시도 (timeout은 1초이므로 반드시 실패해야 함)
        print("[Main] 락 획득을 시도합니다 (timeout=1.0s)...")
        start_time = time.time()

        with pytest.raises(SessionLockTimeoutError) as cm:
            with manager._acquire_lock(manager):
                # 여기는 도달하지 않아야 함
                pass

        duration = time.time() - start_time
        print(f"[Main] 기대한 대로 예외 발생! (소요 시간: {duration:.2f}s)")
        print(f"[Main] 예외 메시지: {cm.value.message}")
        print(f"[Main] 상세 정보: {cm.value.details}")

        assert duration >= 1.0
        assert duration < 2.0

        t1.join()


if __name__ == "__main__":
    unittest.main()
