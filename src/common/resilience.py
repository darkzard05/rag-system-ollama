"""
공통 복원력(resilience) 단일 진원 (Phase 1A 병합).

구 ``common.retry`` (지수 백오프 재시도 + 스트림 재시도)와
구 ``common.circuit_breaker`` (서킷 브레이커 상태 머신)를 흡수했다.
내부 로직은 변경 없이 이동했으며, 구경로는 순수 re-export shim으로 유지한다
(Phase 1 only, Task10에서 제거).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")
# 동기 콜러블이든 비동기 콜러블이든 수용하기 위한 폭넓은 시그니처 별칭.
_Retryable = Callable[..., Any]

# retry_stream 이 재시도하는 전송/네트워크 오류 집합.
_STREAM_RETRY_ON: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    httpx.RequestError,
    httpx.TimeoutException,
)


def retry_with_backoff(
    fn: _Retryable,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    use_async_sleep: bool = True,
) -> Any:
    """값을 반환하는 콜러블에 지수 백오프 재시도를 적용합니다.

    동기 함수면 ``time.sleep``을, 비동기 함수면 ``asyncio.sleep``을 사용합니다.
    (동기 경로는 항상 ``time.sleep`` — ``use_async_sleep``은 비동기 경로에만 영향.)

    Args:
        fn: 재시도할 동기/비동기 콜러블.
        max_retries: 최대 시도 횟수 (1이면 재시도 없음).
        base_delay: 첫 백오프 지연(초).
        backoff: 지수 계수.
        retry_on: 재시도 대상 예외 튜플.
        use_async_sleep: 비동기 경로에서 ``asyncio.sleep``을 사용할지 여부
            (기본 True; False면 ``time.sleep`` 사용). 동기 경로는 항상
            ``time.sleep``을 사용합니다.

    Returns:
        동기 fn이면 반환값(T), 비동기 fn이면 awaitable(Awaitable[T]).
    """
    if asyncio.iscoroutinefunction(fn):
        return _retry_async(
            fn, max_retries, base_delay, backoff, retry_on, use_async_sleep
        )
    return _retry_sync(fn, max_retries, base_delay, backoff, retry_on, use_async_sleep)


def _retry_sync(
    fn: _Retryable,
    max_retries: int,
    base_delay: float,
    backoff: float,
    retry_on: tuple[type[Exception], ...],
    use_async_sleep: bool,  # noqa: ARG001 - sync 경로는 항상 time.sleep
) -> Any:
    for attempt in range(max_retries):
        try:
            return fn()
        except retry_on as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (backoff**attempt)
            logger.warning(
                f"[RETRY] Sync retry {attempt + 1}/{max_retries} "
                f"after {delay:.1f}s: {exc}"
            )
            time.sleep(delay)
    # max_retries == 0 같은 방어 코드 경로; 실제로는 위 루프에서 반환/예외.
    raise RuntimeError("unreachable: retry loop terminated without result")


async def _retry_async(
    fn: _Retryable,
    max_retries: int,
    base_delay: float,
    backoff: float,
    retry_on: tuple[type[Exception], ...],
    use_async_sleep: bool,
) -> Any:
    for attempt in range(max_retries):
        try:
            return await fn()
        except retry_on as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (backoff**attempt)
            logger.warning(
                f"[RETRY] Async retry {attempt + 1}/{max_retries} "
                f"after {delay:.1f}s: {exc}"
            )
            if use_async_sleep:
                await asyncio.sleep(delay)
            else:
                time.sleep(delay)
    raise RuntimeError("unreachable: retry loop terminated without result")


async def retry_stream(
    event_stream_factory: Callable[[], AsyncIterator[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> AsyncIterator[T]:
    """비동기 제너레이터에 지수 백오프 재시도를 적용합니다.

    ``_stream_with_retry``(rag_core.py) 의미론을 보존합니다:

    - ``async for item in event_stream_factory(): yield item``
    - 첫 ``yield`` 시 ``yielded_any`` 를 True로 설정.
    - ``ConnectionError/TimeoutError/OSError/httpx.RequestError/``
      ``httpx.TimeoutException`` 발생 시:
        - ``yielded_any`` 가 True 면 재시도하지 않고 재발생 (중복 전송 방지).
        - 아니면 ``base_delay * 2**attempt`` 대기 후 재시도.
    - 마지막 시도(``attempt == max_retries - 1``)의 오류는 재시도 없이 재발생.
    - ``asyncio.CancelledError`` 는 항상 재발생.

    Args:
        event_stream_factory: 호출 시 ``AsyncIterator[T]`` 를 반환하는 팩토리.
        max_retries: 최대 시도 횟수.
        base_delay: 첫 백오프 지연(초).

    Yields:
        원본 스트림이 내보내는 항목 (T).
    """
    for attempt in range(max_retries):
        yielded_any = False
        try:
            async for item in event_stream_factory():
                yielded_any = True
                yield item
            return
        except _STREAM_RETRY_ON as exc:
            if yielded_any:
                # 첫 토큰 이후 오류는 재시도하지 않는다 (중복 전송 방지).
                raise
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                f"[RETRY] Stream retry {attempt + 1}/{max_retries} "
                f"after {delay:.1f}s: {exc}"
            )
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise


class CircuitBreakerOpen(Exception):
    """서킷 브레이커 열림"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__(f"{service_name} 서킷 브레이커가 열려있습니다")


class CircuitBreakerState(Enum):
    """서킷 브레이커 상태"""

    CLOSED = "closed"  # 정상 (요청 통과)
    OPEN = "open"  # 차단 (요청 거절)
    HALF_OPEN = "half_open"  # 회복중 (일부 요청 허용)


@dataclass
class CircuitBreakerMetrics:
    """서킷 브레이커 메트릭"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """실패율"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class CircuitBreaker:
    """
    서킷 브레이커 패턴 구현

    상태 전이:
    CLOSED (정상)
      ↓ (실패 임계값 초과)
    OPEN (차단)
      ↓ (타임아웃)
    HALF_OPEN (회복중)
      ↓ (성공/실패)
    CLOSED/OPEN
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        """
        Args:
            service_name: 서비스 이름
            failure_threshold: OPEN 상태 전환 실패 횟수
            recovery_timeout: HALF_OPEN 상태까지의 타임아웃 (초)
            success_threshold: HALF_OPEN에서 성공으로 CLOSED 전환 필요 횟수
        """
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.last_state_change_time = datetime.now()

        self.metrics = CircuitBreakerMetrics()
        self.lock = RLock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        서킷 브레이커로 보호된 함수 호출 (동기)

        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자

        Returns:
            함수 반환값

        Raises:
            CircuitBreakerOpen: 서킷이 OPEN 상태일 때
        """
        with self.lock:
            # 상태 확인
            if self.state == CircuitBreakerState.OPEN:
                # 회복 타임아웃 확인
                if self._should_attempt_reset():
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                else:
                    # 여전히 OPEN 상태
                    self.metrics.rejected_requests += 1
                    raise CircuitBreakerOpen(self.service_name)

            # 요청 실행
            try:
                self.metrics.total_requests += 1
                result = func(*args, **kwargs)
                self._on_success()
                return result

            except Exception:
                self._on_failure()
                raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        서킷 브레이커로 보호된 async 함수 호출

        Args:
            func: 실행할 async 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자

        Returns:
            함수 반환값

        Raises:
            CircuitBreakerOpen: 서킷이 OPEN 상태일 때
        """
        with self.lock:
            # 상태 확인
            if self.state == CircuitBreakerState.OPEN:
                # 회복 타임아웃 확인
                if self._should_attempt_reset():
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                else:
                    # 여전히 OPEN 상태
                    self.metrics.rejected_requests += 1
                    logger.warning(f"[CircuitBreaker] {self.service_name} 차단 상태")
                    raise CircuitBreakerOpen(self.service_name)

            # 요청 실행
            try:
                self.metrics.total_requests += 1
                result = await func(*args, **kwargs)
                self._on_success()
                return result

            except Exception:
                self._on_failure()
                raise

    async def call_async_stream(self, func: Callable, *args, **kwargs) -> AsyncIterator:
        """
        서킷 브레이커로 보호된 async generator 호출.
        call_async 대신 async for로 소비해야 하는 함수에 사용합니다.
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                else:
                    self.metrics.rejected_requests += 1
                    logger.warning(f"[CircuitBreaker] {self.service_name} 차단 상태")
                    raise CircuitBreakerOpen(self.service_name)

        self.metrics.total_requests += 1
        try:
            async for item in func(*args, **kwargs):
                yield item
            self._on_success()
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """성공 처리"""
        with self.lock:
            self.metrics.successful_requests += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1

                # HALF_OPEN에서 CLOSED로 전환
                if self.success_count >= self.success_threshold:
                    logger.info(
                        f"[CircuitBreaker] {self.service_name} 회복됨 "
                        f"(성공 {self.success_count}회)"
                    )
                    self._change_state(CircuitBreakerState.CLOSED)

            elif self.state == CircuitBreakerState.CLOSED:
                # 실패 카운트 리셋
                self.failure_count = 0

    def _on_failure(self) -> None:
        """실패 처리"""
        with self.lock:
            self.metrics.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitBreakerState.HALF_OPEN:
                # HALF_OPEN에서 OPEN으로 즉시 전환
                logger.warning(
                    f"[CircuitBreaker] {self.service_name} 회복 실패, 다시 차단 상태로"
                )
                self._change_state(CircuitBreakerState.OPEN)
                self.failure_count = 1  # 실패 카운트 리셋

            elif (
                self.state == CircuitBreakerState.CLOSED
                and self.failure_count >= self.failure_threshold
            ):
                # CLOSED에서 OPEN으로 전환
                logger.warning(
                    f"[CircuitBreaker] {self.service_name} 실패 임계값 초과 "
                    f"({self.failure_count}/{self.failure_threshold}), 차단 상태로"
                )
                self._change_state(CircuitBreakerState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """회복 시도 여부"""
        if self.last_failure_time is None:
            return False

        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout

    def _change_state(self, new_state: CircuitBreakerState) -> None:
        """상태 변경"""
        old_state = self.state
        self.state = new_state
        self.last_state_change_time = datetime.now()

        logger.info(
            f"[CircuitBreaker] {self.service_name} 상태 변경: "
            f"{old_state.value} → {new_state.value}"
        )

        # 상태 변경 기록
        self.metrics.state_changes.append(f"{old_state.value} → {new_state.value}")

        # 상태별 초기화
        if new_state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self.success_count = 0

    def get_state(self) -> str:
        """현재 상태 반환"""
        with self.lock:
            return self.state.value

    def get_metrics(self) -> dict[str, Any]:
        """메트릭 조회"""
        with self.lock:
            return {
                "service_name": self.service_name,
                "state": self.state.value,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "rejected_requests": self.metrics.rejected_requests,
                "failure_rate": f"{self.metrics.failure_rate:.1%}",
                "success_rate": f"{self.metrics.success_rate:.1%}",
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat()
                if self.last_failure_time
                else None,
                "last_state_change": self.last_state_change_time.isoformat(),
            }

    def reset(self) -> None:
        """서킷 브레이커 리셋"""
        with self.lock:
            logger.info(f"[CircuitBreaker] {self.service_name} 리셋")
            self._change_state(CircuitBreakerState.CLOSED)
            self.failure_count = 0
            self.success_count = 0


class CircuitBreakerRegistry:
    """
    여러 서킷 브레이커 관리

    지원 모드:
    1. Global: 모든 사용자 공유 (예: 외부 API 리미트)
    2. Session: 세션별 독립 (예: 사용자별 오동작 방지)
    """

    def __init__(self):
        self.global_breakers: dict[str, CircuitBreaker] = {}
        self.session_breakers: dict[str, dict[str, CircuitBreaker]] = {}
        self.lock = RLock()

    def get_breaker(
        self, service_name: str, session_id: str | None = None, **kwargs
    ) -> CircuitBreaker:
        """
        서킷 브레이커 획득

        session_id가 제공되면 해당 세션 전용 브레이커를 반환하고,
        없으면 전역 브레이커를 반환합니다.
        """
        with self.lock:
            if session_id:
                if session_id not in self.session_breakers:
                    self.session_breakers[session_id] = {}

                if service_name not in self.session_breakers[session_id]:
                    self.session_breakers[session_id][service_name] = CircuitBreaker(
                        service_name=f"{service_name}:{session_id}", **kwargs
                    )
                return self.session_breakers[session_id][service_name]
            else:
                if service_name not in self.global_breakers:
                    self.global_breakers[service_name] = CircuitBreaker(
                        service_name=service_name, **kwargs
                    )
                return self.global_breakers[service_name]


# 전역 레지스트리

_circuit_breaker_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """서킷 브레이커 레지스트리 반환"""
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    return _circuit_breaker_registry


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerMetrics",
    "CircuitBreakerOpen",
    "CircuitBreakerRegistry",
    "CircuitBreakerState",
    "get_circuit_breaker_registry",
    "retry_stream",
    "retry_with_backoff",
]
