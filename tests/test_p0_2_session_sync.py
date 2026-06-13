"""
P0-2 결함 테스트: 세션 상태 동기화 경합 조건

테스트 목표:
1. 다중 스레드 동시 접근 안전성
2. 루프 변경 감지 및 복구
3. Streamlit 상태 동기화 정확성
"""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from core.session import SessionManager


class TestSessionSynchronization:
    """세션 상태 동기화 테스트 클래스"""

    def test_2_1_concurrent_state_access(self):
        """테스트 2.1: 10개 스레드에서 동시에 상태 접근"""
        SessionManager.init_session("test_concurrent")

        results = {"errors": [], "success_count": 0}

        def worker(thread_id):
            try:
                for i in range(50):
                    SessionManager.set(
                        f"key_{thread_id}_{i}",
                        f"value_{i}",
                        session_id="test_concurrent"
                    )
                    value = SessionManager.get(
                        f"key_{thread_id}_{i}",
                        session_id="test_concurrent"
                    )
                    assert value == f"value_{i}", f"값 불일치: {value}"
                    results["success_count"] += 1
            except Exception as e:
                results["errors"].append((thread_id, str(e)))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()

        assert len(results["errors"]) == 0, f"경합 오류: {results['errors']}"
        assert results["success_count"] == 500
        print(f"✅ 테스트 2.1: 500회 동시 접근 성공 (오류: {len(results['errors'])}건)")

    def test_2_2_message_consistency(self):
        """테스트 2.2: 메시지 추가 시 데이터 정합성"""
        SessionManager.init_session("test_messages")

        # 100개 메시지 동시 추가
        def add_messages():
            for i in range(50):
                SessionManager.add_message(
                    role="user",
                    content=f"Message {i}",
                    session_id="test_messages"
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(add_messages) for _ in range(2)]
            for future in futures:
                future.result()

        messages = SessionManager.get_messages(session_id="test_messages")
        # 최대 100개 저장 (초과분 제거)
        assert len(messages) <= 100, f"메시지 초과: {len(messages)}"
        assert all("content" in m for m in messages), "메시지 필드 누락"
        print(f"✅ 테스트 2.2: 메시지 정합성 확인 ({len(messages)}개 메시지)")

    def test_2_3_status_log_consistency(self):
        """테스트 2.3: 상태 로그 추가 시 정합성"""
        SessionManager.init_session("test_logs")

        # 100개 로그 추가
        for i in range(100):
            SessionManager.add_status_log(
                f"Status {i}",
                session_id="test_logs",
                add_to_chat=False  # 메시지 이중 추가 방지
            )

        logs = SessionManager.get("status_logs", session_id="test_logs")
        # 최대 30개 저장 (마지막 30개만)
        assert len(logs) <= 30, f"로그 초과: {len(logs)}"
        print(f"✅ 테스트 2.3: 상태 로그 정합성 확인 ({len(logs)}개 로그)")

    def test_2_4_session_cleanup(self):
        """테스트 2.4: 세션 정리 시 리소스 해제"""
        SessionManager.init_session("test_cleanup")

        # 상태 설정
        SessionManager.set("test_key", "test_value", session_id="test_cleanup")
        SessionManager.set("rag_engine", "mock_engine", session_id="test_cleanup")

        # 세션 삭제
        result = SessionManager.delete_session("test_cleanup")
        assert result, "세션 삭제 실패"

        # 삭제된 세션에서 데이터 조회 (기본값 반환)
        value = SessionManager.get(
            "test_key",
            default="NOT_FOUND",
            session_id="test_cleanup",
            create=False
        )
        assert value == "NOT_FOUND", "삭제된 세션 데이터 여전히 존재"
        print("✅ 테스트 2.4: 세션 정리 및 리소스 해제 완료")

    def test_2_5_lock_efficiency(self):
        """테스트 2.5: 락 경합 최소화 (성능 검증)"""
        import time

        SessionManager.init_session("test_lock")

        # 1초 내에 1000개 작업 완료 (락 효율성)
        start = time.time()
        for i in range(1000):
            SessionManager.set(
                f"perf_key_{i}",
                f"value_{i}",
                session_id="test_lock"
            )
        elapsed = time.time() - start

        # 1000개 작업이 1초 이내에 완료 (락 경합 최소)
        assert elapsed < 1.0, f"락 경합 의심: {elapsed:.2f}초 소요"
        print(f"✅ 테스트 2.5: 락 효율성 확인 (1000개 작업 {elapsed:.3f}초)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
