"""
프로젝트 전반에서 사용되는 유틸리티 함수들을 모아놓은 파일.
"""

import time
import logging
import functools
import asyncio
import nest_asyncio
import re


logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    텍스트를 정제합니다.
    1. 널 문자 제거
    2. 연속된 공백/탭/개행을 단일 공백으로 정규화
    """
    if not text:
        return ""
    # 널 문자 제거
    text = text.replace("\x00", "")
    # 연속된 공백을 단일 공백으로 치환
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def sync_run(coro):
    """
    코루틴을 동기적으로 실행하는 안전한 래퍼 함수입니다.
    
    이미 실행 중인 이벤트 루프가 있으면 nest_asyncio를 활용해 해당 루프에서 실행하고,
    그렇지 않으면 새로운 asyncio.run()으로 실행합니다.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 이미 루프가 실행 중인 경우 (Streamlit 등)
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    else:
        # 실행 중인 루프가 없는 경우
        return asyncio.run(coro)


def log_operation(operation_name):
    """
    함수 실행 시작, 완료, 오류 발생 시 로그를 남기는 데코레이터.
    실행 시간도 함께 기록합니다. (동기 함수용)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"[작업 시작] '{operation_name}'")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"[작업 완료] '{operation_name}' (소요시간: {duration:.2f}초)"
                )
                return result
            except Exception as e:
                logger.error(
                    f"[작업 실패] '{operation_name}' - 오류: {e}", exc_info=True
                )
                raise

        return wrapper

    return decorator


def async_log_operation(operation_name):
    """
    비동기 함수 및 비동기 생성기의 실행 시작, 완료, 오류 발생 시 로그를 남기는 데코레이터.
    스트리밍(yield)을 지원하면서 전체 실행 시간을 정확히 측정합니다.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger.info(f"[작업 시작] '{operation_name}'")
            start_time = time.time()
            try:
                # 비동기 생성기를 호출하고, 그 결과를 스트리밍합니다.
                async for chunk in func(*args, **kwargs):
                    yield chunk
            except Exception as e:
                logger.error(
                    f"[작업 실패] '{operation_name}' - 오류: {e}", exc_info=True
                )
                raise
            finally:
                # 스트리밍이 모두 끝난 후에 완료 로그를 남깁니다.
                duration = time.time() - start_time
                logger.info(
                    f"[작업 완료] '{operation_name}' (소요시간: {duration:.2f}초)"
                )

        return wrapper

    return decorator
