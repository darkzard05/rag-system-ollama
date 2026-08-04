"""
Comprehensive thread safety tests for SessionManager.

Tests for:
- Concurrent read/write operations
- Race condition detection (atomic read-modify-write)
- Deadlock prevention
- Per-session lock isolation
- Atomic multi-key updates
"""

import logging
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.session import SessionManager  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# Basic Operation Tests
# ============================================================================


class TestBasicOperations(unittest.TestCase):
    """Test basic session operations."""

    def setUp(self):
        SessionManager.reset()

    def test_get_with_default(self):
        """Test get with default value."""
        SessionManager.set_session_id("basic_default")
        assert SessionManager.get("nonexistent", default="default_value") == (
            "default_value"
        )

    def test_multi_key_set(self):
        """Test atomic multi-key update via kwargs."""
        SessionManager.set_session_id("basic_multi")
        SessionManager.set(key1="value1", key2="value2")
        assert SessionManager.get("key1") == "value1"
        assert SessionManager.get("key2") == "value2"

    def test_message_accumulation(self):
        """Test message and status log accumulation."""
        SessionManager.set_session_id("basic_msgs")
        SessionManager.add_message("user", "hello")
        SessionManager.add_status_log("로그 기록")
        assert len(SessionManager.get_messages()) == 1
        assert "로그 기록" in SessionManager.get("status_logs")

    def test_delete_session(self):
        """Test delete_session removes the session."""
        SessionManager.init_session(session_id="basic_clear")
        SessionManager.set("key1", "value1", session_id="basic_clear")
        assert SessionManager.delete_session("basic_clear")
        assert not SessionManager.delete_session("basic_clear")


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess(unittest.TestCase):
    """Test thread-safe concurrent access."""

    def setUp(self):
        SessionManager.reset()

    def test_concurrent_writes(self):
        """Test concurrent write operations."""
        sid = "conc_write"
        SessionManager.init_session(session_id=sid)

        def write_value(i: int):
            SessionManager.set(f"key_{i}", f"value_{i}", session_id=sid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_value, i) for i in range(100)]
            for future in as_completed(futures):
                future.result()

        # Verify all values were written
        for i in range(100):
            value = SessionManager.get(f"key_{i}", session_id=sid)
            assert value == f"value_{i}", f"key_{i} 값 손실: {value}"

    def test_concurrent_reads(self):
        """Test concurrent read operations."""
        sid = "conc_read"
        SessionManager.set("shared_key", "shared_value", session_id=sid)

        results = []

        def read_value():
            results.append(SessionManager.get("shared_key", session_id=sid))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(read_value) for _ in range(200)]
            for future in as_completed(futures):
                future.result()

        # All reads should return same value
        assert len(results) == 200
        assert all(v == "shared_value" for v in results)

    def test_concurrent_add_message(self):
        """Test concurrent add_message (atomic read-modify-write).

        add_message은 세션별 락 아래에서 append를 수행하므로
        동시에 50개를 추가해도 하나도 유실되지 않아야 합니다.
        """
        sid = "conc_msg"
        SessionManager.init_session(session_id=sid)

        def add_message(i: int):
            SessionManager.add_message("user", f"msg {i}", session_id=sid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_message, i) for i in range(50)]
            for future in as_completed(futures):
                future.result()

        messages = SessionManager.get_messages(session_id=sid)
        assert len(messages) == 50, f"메시지 유실: {len(messages)}/50"

    def test_concurrent_read_write_mix(self):
        """Test mixed concurrent read and write operations."""
        sid = "conc_mix"
        for i in range(10):
            SessionManager.set(f"key_{i}", i, session_id=sid)

        read_count = [0]
        write_count = [0]

        def mixed_operation(idx: int):
            if idx % 2 == 0:
                # Read operation
                SessionManager.get(f"key_{idx % 10}", session_id=sid)
                read_count[0] += 1
            else:
                # Write operation
                SessionManager.set(f"key_{idx % 10}", idx, session_id=sid)
                write_count[0] += 1

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(mixed_operation, i) for i in range(300)]
            for future in as_completed(futures):
                future.result()

        logger.info(
            "Mixed operations verified (%d reads, %d writes)",
            read_count[0],
            write_count[0],
        )

    def test_concurrent_delete(self):
        """Test concurrent delete operations."""
        sid = "conc_del"
        for i in range(50):
            SessionManager.set(f"delete_key_{i}", f"value_{i}", session_id=sid)

        def delete_value(i: int):
            SessionManager.delete(f"delete_key_{i}", session_id=sid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(delete_value, i) for i in range(50)]
            for future in as_completed(futures):
                future.result()

        # Verify all values were deleted
        for i in range(50):
            assert SessionManager.get(f"delete_key_{i}", session_id=sid) is None


# ============================================================================
# Race Condition Tests
# ============================================================================


class TestRaceConditions(unittest.TestCase):
    """Test race condition prevention."""

    def setUp(self):
        SessionManager.reset()

    def test_shared_lock_identity(self):
        """같은 세션은 동일한 락을 공유하고 다른 세션은 별도의 락을 가집니다."""
        lock_a1 = SessionManager._acquire_lock("race_sid_a")
        lock_a2 = SessionManager._acquire_lock("race_sid_a")
        lock_b = SessionManager._acquire_lock("race_sid_b")

        assert lock_a1 is lock_a2
        assert lock_a1 is not lock_b

    def test_get_set_race_condition(self):
        """get+set 조합은 원자적이지 않음 (약한 테스트).

        신규 SessionManager의 원자적 연산은 add_message/add_status_log와
        같은 내부 락 기반 read-modify-write입니다.
        """
        sid = "race_counter"
        SessionManager.set("counter", 0, session_id=sid)

        def increment():
            current = SessionManager.get("counter", 0, session_id=sid)
            time.sleep(0.0001)  # Create race window
            SessionManager.set("counter", current + 1, session_id=sid)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(increment) for _ in range(100)]
            for future in as_completed(futures):
                future.result()

        # Note: get+set은 원자적이지 않으므로 최종 값은 100보다 작을 수 있음
        final_count = SessionManager.get("counter", 0, session_id=sid)
        logger.info("Counter race condition test: final count = %d/100", final_count)

    def test_atomic_add_status_log(self):
        """add_status_log는 세션별 락 아래에서 원자적으로 수행됩니다.

        (status_logs는 최대 30개로 유지되므로 캡 미만인 30개를 동시 추가)
        """
        sid = "race_log"
        SessionManager.init_session(session_id=sid)

        def add_log(i: int):
            SessionManager.add_status_log(f"log {i}", session_id=sid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_log, i) for i in range(30)]
            for future in as_completed(futures):
                future.result()

        logs = SessionManager.get("status_logs", session_id=sid)
        assert len(logs) == 30, f"상태 로그 유실: {len(logs)}/30"


# ============================================================================
# Deadlock Prevention Tests
# ============================================================================


class TestDeadlockPrevention(unittest.TestCase):
    """Test deadlock prevention mechanisms."""

    def setUp(self):
        SessionManager.reset()

    def test_no_deadlock_in_sequential_operations(self):
        """동일 세션에 대한 get/set 연속 호출이 데드락을 일으키지 않아야 합니다."""
        sid = "deadlock_seq"
        SessionManager.set("test_key", "initial_value", session_id=sid)

        def sequential_operation():
            value1 = SessionManager.get("test_key", session_id=sid)
            SessionManager.set("test_key", value1 + "_modified", session_id=sid)
            value2 = SessionManager.get("test_key", session_id=sid)
            assert "_modified" in value2

        thread = threading.Thread(target=sequential_operation)
        thread.start()
        thread.join(timeout=5.0)

        assert not thread.is_alive(), "Thread should complete (no deadlock)"

    def test_no_deadlock_under_lock_contention(self):
        """System remains responsive under high lock contention."""
        sid = "deadlock_contention"
        start_time = time.time()

        def heavy_contention():
            for _ in range(100):
                SessionManager.set("contested_key", "value", session_id=sid)
                SessionManager.get("contested_key", session_id=sid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(heavy_contention) for _ in range(10)]
            for future in as_completed(futures):
                future.result()

        elapsed = time.time() - start_time
        assert elapsed < 10.0, "High contention should complete within 10 seconds"
        logger.info("No deadlock under contention (%.2fs)", elapsed)

    def test_per_session_lock_isolation(self):
        """다른 세션의 락은 서로 블록하지 않습니다."""
        sid_a = "deadlock_iso_a"
        sid_b = "deadlock_iso_b"

        def hold_lock(session_id: str, hold_seconds: float):
            SessionManager.init_session(session_id=session_id)
            with SessionManager._acquire_lock(session_id):
                time.sleep(hold_seconds)

        start_time = time.time()
        t1 = threading.Thread(target=hold_lock, args=(sid_a, 0.5))
        t2 = threading.Thread(target=hold_lock, args=(sid_b, 0.1))

        t1.start()
        time.sleep(0.05)
        t2.start()

        t1.join(timeout=5)
        t2.join(timeout=5)
        elapsed = time.time() - start_time

        assert not t1.is_alive()
        assert not t2.is_alive()
        # 전역 락이었다면 0.6초 이상, 세션별 락이면 약 0.5초 이내
        assert elapsed < 0.6


# ============================================================================
# Statistics and Monitoring Tests
# ============================================================================


class TestStatistics(unittest.TestCase):
    """Test statistics and monitoring."""

    def setUp(self):
        SessionManager.reset()

    def test_statistics_tracking(self):
        """Test statistics are properly tracked."""
        sid = "stats"
        SessionManager.init_session(session_id=sid)
        SessionManager.set("stat_key", "value", session_id=sid)
        SessionManager.add_message("user", "hi", session_id=sid)

        stats = SessionManager.get_stats()

        assert stats["active_sessions"] == 1
        assert stats["total_messages"] == 1

    def test_stats_after_delete_session(self):
        """Test statistics reflect session deletion."""
        sid = "stats_del"
        SessionManager.init_session(session_id=sid)
        SessionManager.set("key", "value", session_id=sid)
        SessionManager.delete_session(sid)

        stats = SessionManager.get_stats()
        assert stats["active_sessions"] == 0


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestSessionScopedHelpers(unittest.TestCase):
    """세션 ID 기반 헬퍼 함수 (API 계층 사용 방식 모사)."""

    SID = "helper_sid"

    def setUp(self):
        SessionManager.reset()
        SessionManager.init_session(session_id=self.SID)

    def ts_get(self, key, default=None):
        return SessionManager.get(key, default, session_id=self.SID)

    def ts_set(self, key, value):
        return SessionManager.set(key, value, session_id=self.SID)

    def ts_delete(self, key):
        return SessionManager.delete(key, session_id=self.SID)

    def test_helper_set_get(self):
        """Test session-scoped helpers for set/get."""
        self.ts_set("conv_key", "conv_value")
        assert self.ts_get("conv_key") == "conv_value"

    def test_helper_delete(self):
        """Test session-scoped helper for delete."""
        self.ts_set("del_key", "value")
        self.ts_delete("del_key")
        assert self.ts_get("del_key") is None

    def test_helper_default(self):
        """Test session-scoped helper with default value."""
        assert self.ts_get("nonexistent", "default_value") == "default_value"
