"""
CircuitBreaker 상태전이 단위 테스트.

src/common/circuit_breaker.py 의 상태 머신(CLOSED -> OPEN -> HALF_OPEN ->
CLOSED/OPEN) 전 분기와 메트릭 집계를 순수 로직으로 검증한다.
외부 의존성 없음(모델/네트워크 미사용). 시간 제어는 datetime 주입으로 수행.

참고: call() 이 임계값을 넘겨 OPEN 으로 전이할 때도 원본 예외(RuntimeError
등)를 그대로 전파한다(CircuitBreakerOpen 아님). 따라서 OPEN 강제는 상태를
직접 할당하거나, 임계 호출은 원본 예외로 잡는다.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta

import pytest

from common.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerState,
)


def _fail() -> None:
    raise RuntimeError("boom")


def _ok() -> int:
    return 42


async def _ok_async() -> int:
    return 7


async def _fail_async() -> int:
    raise RuntimeError("async boom")


async def _gen() -> AsyncGenerator[int, None]:
    for i in range(3):
        yield i


def _force_open(cb: CircuitBreaker) -> None:
    """상태를 OPEN 으로 강제 전환(임계/타임아웃과 무관)."""
    cb.state = CircuitBreakerState.OPEN
    cb.last_failure_time = datetime.now()


def test_initial_state_closed() -> None:
    """신규 인스턴스는 CLOSED, 실패 카운트 0."""
    cb = CircuitBreaker(service_name="svc")
    assert cb.get_state() == CircuitBreakerState.CLOSED.value
    assert cb.failure_count == 0


def test_closed_to_open_on_threshold() -> None:
    """failure_threshold 도달 시 CLOSED -> OPEN 전이(원본 예외 전파)."""
    cb = CircuitBreaker(service_name="svc", failure_threshold=3)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(_fail)
    assert cb.get_state() == CircuitBreakerState.CLOSED.value

    # 3번째 실패: 임계 도달 -> OPEN 전이. 원본 예외는 그대로 전파.
    with pytest.raises(RuntimeError):
        cb.call(_fail)
    assert cb.get_state() == CircuitBreakerState.OPEN.value
    assert "closed → open" in cb.metrics.state_changes


def test_closed_success_resets_failure_count() -> None:
    """CLOSED 상태에서 성공 시 실패 카운트 리셋."""
    cb = CircuitBreaker(service_name="svc", failure_threshold=3)
    with pytest.raises(RuntimeError):
        cb.call(_fail)
    with pytest.raises(RuntimeError):
        cb.call(_fail)
    assert cb.call(_ok) == 42
    assert cb.failure_count == 0
    assert cb.get_state() == CircuitBreakerState.CLOSED.value


def test_open_rejects_immediately() -> None:
    """OPEN 상태에서 내부 함수 호출 없이 즉시 CircuitBreakerOpen."""
    cb = CircuitBreaker(service_name="svc")
    _force_open(cb)
    assert cb.get_state() == CircuitBreakerState.OPEN.value

    calls = 0

    def _tracked() -> int:
        nonlocal calls
        calls += 1
        return 1

    with pytest.raises(CircuitBreakerOpen):
        cb.call(_tracked)
    assert calls == 0
    assert cb.get_metrics()["rejected_requests"] >= 1


def test_open_to_half_open_after_timeout() -> None:
    """recovery_timeout 경과 후 OPEN -> HALF_OPEN 전이."""
    cb = CircuitBreaker(service_name="svc", recovery_timeout=60.0)
    _force_open(cb)

    # 타임아웃 경과 시점으로 주입
    cb.last_failure_time = datetime.now() - timedelta(seconds=cb.recovery_timeout + 1)
    assert cb.call(_ok) == 42
    assert cb.get_state() == CircuitBreakerState.HALF_OPEN.value


def test_half_open_to_closed_on_success_threshold() -> None:
    """HALF_OPEN에서 success_threshold 성공 시 CLOSED 복귀."""
    cb = CircuitBreaker(service_name="svc", recovery_timeout=60.0, success_threshold=2)
    _force_open(cb)
    cb.last_failure_time = datetime.now() - timedelta(seconds=cb.recovery_timeout + 1)
    assert cb.call(_ok) == 42  # -> HALF_OPEN
    assert cb.get_state() == CircuitBreakerState.HALF_OPEN.value

    assert cb.call(_ok) == 42  # success_threshold 도달 -> CLOSED
    assert cb.get_state() == CircuitBreakerState.CLOSED.value
    assert cb.success_count == 0


def test_half_open_to_open_on_failure() -> None:
    """HALF_OPEN에서 1회 실패 시 즉시 OPEN 복귀."""
    cb = CircuitBreaker(service_name="svc", recovery_timeout=60.0, success_threshold=2)
    _force_open(cb)
    cb.last_failure_time = datetime.now() - timedelta(seconds=cb.recovery_timeout + 1)
    assert cb.call(_ok) == 42  # -> HALF_OPEN
    assert cb.get_state() == CircuitBreakerState.HALF_OPEN.value

    with pytest.raises(RuntimeError):
        cb.call(_fail)  # HALF_OPEN 실패 -> OPEN
    assert cb.get_state() == CircuitBreakerState.OPEN.value


@pytest.mark.asyncio
async def test_call_async_open_transition() -> None:
    """call_async 경로에서 OPEN 즉시 거부 검증."""
    cb = CircuitBreaker(service_name="svc")
    _force_open(cb)
    assert cb.get_state() == CircuitBreakerState.OPEN.value

    calls = 0

    async def _tracked() -> int:
        nonlocal calls
        calls += 1
        return 1

    with pytest.raises(CircuitBreakerOpen):
        await cb.call_async(_tracked)
    assert calls == 0
    assert cb.get_metrics()["rejected_requests"] >= 1


@pytest.mark.asyncio
async def test_call_async_success_records_metrics() -> None:
    """call_async 성공 경로가 결과를 반환하고 메트릭을 집계."""
    cb = CircuitBreaker(service_name="svc", failure_threshold=3)
    result = await cb.call_async(_ok_async)
    assert result == 7
    assert cb.get_state() == CircuitBreakerState.CLOSED.value
    m = cb.get_metrics()
    assert m["total_requests"] == 1
    assert m["successful_requests"] == 1


def test_metrics_failure_rate() -> None:
    """failure_rate / success_rate 산술 검증."""
    cb = CircuitBreaker(service_name="svc", failure_threshold=10)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(_fail)
    assert cb.call(_ok) == 42
    assert cb.call(_ok) == 42
    m = cb.get_metrics()
    assert m["total_requests"] == 4
    assert m["failed_requests"] == 2
    assert m["successful_requests"] == 2
    assert m["failure_rate"] == "50.0%"
    assert m["success_rate"] == "50.0%"


def test_reset_returns_to_closed() -> None:
    """reset() 은 어떤 상태에서든 CLOSED + 카운트 리셋."""
    cb = CircuitBreaker(service_name="svc")
    _force_open(cb)
    assert cb.get_state() == CircuitBreakerState.OPEN.value

    cb.reset()
    assert cb.get_state() == CircuitBreakerState.CLOSED.value
    assert cb.failure_count == 0
    assert cb.success_count == 0


@pytest.mark.asyncio
async def test_call_async_stream_basic() -> None:
    """CLOSED 상태에서 async generator 보호 호출이 아이템을 그대로 통과."""
    cb = CircuitBreaker(service_name="svc")

    collected = [i async for i in cb.call_async_stream(_gen)]
    assert collected == [0, 1, 2]
    assert cb.get_state() == CircuitBreakerState.CLOSED.value
