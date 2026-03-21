"""
에러 복구 - Task 14
재시도 로직, 지수 백오프, 타임아웃 관리
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RetryStrategy(Enum):
    """재시도 전략"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """재시도 설정"""

    max_attempts: int = 3
    initial_delay: float = 1.0  # 초
    max_delay: float = 60.0  # 초
    exponential_base: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True  # 지터 추가 (동시 재시도 방지)
    retryable_exceptions: list[type[Exception]] = field(
        default_factory=lambda: [
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            IOError,
        ]
    )


class ExponentialBackoff:
    """
    지수 백오프 구현

    재시도 대기 시간:
    - 1차: 1초
    - 2차: 2초
    - 3차: 4초
    - ...
    """

    def __init__(self, config: RetryConfig):
        self.config = config

    def calculate_delay(self, attempt: int) -> float:
        """대기 시간 계산"""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.exponential_base**attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.initial_delay
        else:  # IMMEDIATE
            delay = 0

        # 최대 대기 시간 제한
        delay = min(delay, self.config.max_delay)

        # 지터 추가 (동시 재시도 방지)
        if self.config.jitter:
            import random

            jitter = random.uniform(0, delay * 0.1)
            delay += jitter

        return delay


class RetryPolicy:
    """
    재시도 정책

    특정 예외에 대해서만 재시도
    """

    def __init__(self, config: RetryConfig):
        self.config = config
        self.backoff = ExponentialBackoff(config)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """재시도 여부 판단"""
        # 최대 시도 횟수 초과
        if attempt >= self.config.max_attempts:
            return False

        # 재시도 가능한 예외 확인
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.config.retryable_exceptions
        )

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        재시도 로직이 포함된 함수 실행

        Args:
            func: 실행할 함수 (async)
            *args: 함수 인자
            **kwargs: 함수 키워드 인자

        Returns:
            함수 반환값

        Raises:
            MaxRetriesExceeded: 최대 재시도 횟수 초과
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                logger.debug(
                    f"[Retry] 시도 {attempt + 1}/{self.config.max_attempts}: "
                    f"{func.__name__}"
                )

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # 재시도 여부 확인
                if not self.should_retry(e, attempt):
                    logger.error(
                        f"[Retry] 재시도 불가능한 예외: {type(e).__name__}: {e}"
                    )
                    raise

                # 마지막 시도
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"[Retry] 최대 재시도 횟수 초과: {func.__name__}")
                    raise MaxRetriesExceeded(
                        func_name=func.__name__,
                        attempts=self.config.max_attempts,
                        last_error=str(e),
                    ) from e

                # 대기 후 재시도
                delay = self.backoff.calculate_delay(attempt)
                logger.warning(
                    f"[Retry] {delay:.2f}초 후 재시도 ({attempt + 1}차 실패): "
                    f"{type(e).__name__}: {e}"
                )

                await asyncio.sleep(delay)

        # 도달하면 안 되는 부분
        raise last_exception or Exception("Unknown error")


class TimeoutManager:
    """
    타임아웃 관리

    asyncio.wait_for와 유사하지만 더 유연한 제어
    """

    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout

    async def execute_with_timeout(
        self, func: Callable, timeout: float | None = None, *args, **kwargs
    ) -> Any:
        """
        타임아웃이 적용된 함수 실행

        Args:
            func: 실행할 함수
            timeout: 타임아웃 (초)
            *args: 함수 인자
            **kwargs: 함수 키워드 인자

        Returns:
            함수 반환값

        Raises:
            asyncio.TimeoutError: 타임아웃 초과
        """
        timeout_val = timeout or self.default_timeout

        try:
            logger.debug(f"[Timeout] 타임아웃 {timeout_val}초로 실행: {func.__name__}")

            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_val
                )
            else:
                return func(*args, **kwargs)

        except asyncio.TimeoutError:
            logger.error(f"[Timeout] 타임아웃 초과 ({timeout_val}초): {func.__name__}")
            raise


class AdaptiveTimeout:
    """
    적응형 타임아웃

    과거 실행 시간 기반으로 동적 타임아웃 설정.
    최소 하한선을 두어 비정상적인 초고속 실패로 인한 데드락을 방지합니다.
    """

    def __init__(
        self,
        initial_timeout: float = 30.0,
        min_timeout: float = 1.0,
        percentile: float = 0.95,
    ):
        self.initial_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.percentile = percentile  # p95 사용
        self.execution_times: list[float] = []
        self.max_history = 100

    def record_execution_time(self, duration: float) -> None:
        """실행 시간 기록"""
        self.execution_times.append(duration)

        # 히스토리 크기 제한
        if len(self.execution_times) > self.max_history:
            self.execution_times.pop(0)

    def get_adaptive_timeout(self) -> float:
        """적응형 타임아웃 계산 (하한선 보장)"""
        if not self.execution_times:
            return self.initial_timeout

        # 정렬 후 상위 percentile 값 선택
        sorted_times = sorted(self.execution_times)
        index = int(len(sorted_times) * self.percentile)
        index = min(index, len(sorted_times) - 1)

        p_time = sorted_times[index]

        # 마진 추가 (20%) 및 하한선 보장
        adaptive_timeout = max(self.min_timeout, p_time * 1.2)

        logger.debug(
            f"[AdaptiveTimeout] p{int(self.percentile * 100)}: {p_time:.2f}s, "
            f"최종 적응형: {adaptive_timeout:.2f}s (하한선: {self.min_timeout}s)"
        )

        return adaptive_timeout


@dataclass
class ErrorContext:
    """에러 컨텍스트"""

    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    attempt: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "attempt": self.attempt,
            "metadata": self.metadata,
        }


class ErrorRecoveryChain:
    """
    에러 복구 체인

    여러 복구 전략을 순차적으로 시도
    """

    def __init__(self):
        self.strategies: list[Callable] = []
        self.error_history: list[ErrorContext] = []

    def add_recovery_strategy(self, strategy: Callable) -> None:
        """복구 전략 추가"""
        self.strategies.append(strategy)

    async def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """
        복구 전략과 함께 함수 실행

        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자

        Returns:
            함수 반환값 또는 복구 결과

        Raises:
            Exception: 모든 복구 전략 실패 시
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            logger.error(f"[ErrorRecovery] 주 함수 실패: {type(e).__name__}: {e}")

            # 에러 컨텍스트 기록
            error_ctx = ErrorContext(
                error_type=type(e).__name__,
                error_message=str(e),
                metadata={"func_name": func.__name__},
            )
            self.error_history.append(error_ctx)

            # 복구 전략 시도
            for i, strategy in enumerate(self.strategies):
                try:
                    logger.info(
                        f"[ErrorRecovery] 복구 전략 {i + 1}/{len(self.strategies)} 시도"
                    )

                    if asyncio.iscoroutinefunction(strategy):
                        result = await strategy(e, *args, **kwargs)
                    else:
                        result = strategy(e, *args, **kwargs)

                    logger.info(f"[ErrorRecovery] 복구 전략 {i + 1} 성공")
                    return result

                except Exception as recovery_error:
                    logger.warning(
                        f"[ErrorRecovery] 복구 전략 {i + 1} 실패: {recovery_error}"
                    )
                    continue

            # 모든 복구 전략 실패
            logger.error("[ErrorRecovery] 모든 복구 전략 실패")
            raise

    def get_error_history(self, hours: int = 1) -> list[ErrorContext]:
        """에러 히스토리 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [ctx for ctx in self.error_history if ctx.timestamp > cutoff_time]


# 예외 클래스


class MaxRetriesExceeded(Exception):
    """최대 재시도 횟수 초과"""

    def __init__(self, func_name: str, attempts: int, last_error: str):
        self.func_name = func_name
        self.attempts = attempts
        self.last_error = last_error

        super().__init__(
            f"{func_name}: {attempts}회 재시도 후 실패 (마지막 에러: {last_error})"
        )


class CircuitBreakerOpen(Exception):
    """서킷 브레이커 열림"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__(f"{service_name} 서킷 브레이커가 열려있습니다")


# 전역 재시도 정책

_default_retry_policy: RetryPolicy | None = None
_default_timeout_manager: TimeoutManager | None = None


def get_retry_policy(config: RetryConfig | None = None) -> RetryPolicy:
    """기본 재시도 정책 반환"""
    global _default_retry_policy
    if _default_retry_policy is None:
        _default_retry_policy = RetryPolicy(config or RetryConfig())
    return _default_retry_policy


def get_timeout_manager(default_timeout: float = 30.0) -> TimeoutManager:
    """타임아웃 관리자 반환"""
    global _default_timeout_manager
    if _default_timeout_manager is None:
        _default_timeout_manager = TimeoutManager(default_timeout)
    return _default_timeout_manager


def reset_error_recovery():
    """에러 복구 리셋"""
    global _default_retry_policy, _default_timeout_manager
    _default_retry_policy = None
    _default_timeout_manager = None
