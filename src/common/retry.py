"""
공통 재시도(retry) 유틸리티.

두 가지 형태의 재시도 헬퍼를 제공합니다:

1. ``retry_with_backoff`` — 값을 반환하는 동기/비동기 콜러블용.
   동기 함수는 ``time.sleep``(블로킹)으로, 비동기 함수는 ``asyncio.sleep``으로
   백오프합니다. (R12: 파일 삭제처럼 sync 컨텍스트에서 호출되는 블로킹 경로는
   ``time.sleep``이 정확합니다.)
2. ``retry_stream`` — 비동기 제너레이터용. ``yield``로 항목을 그대로
   전달하며, 첫 항목 전달 이후 발생한 오류는 재시도하지 않고 재발생시켜
   중복 전송을 막습니다. (``_stream_with_retry`` 의미론 보존.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
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
        use_async_sleep: 비동기 경로에서 ``asyncio.sleep`` 대신
            ``time.sleep``을 쓸지 여부 (기본 True).

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
