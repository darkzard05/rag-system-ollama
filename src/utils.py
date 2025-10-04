"""
프로젝트 전반에서 사용되는 유틸리티 함수들을 모아놓은 파일.
"""

import time
import logging
import functools


def log_operation(operation_name):
    """
    함수 실행 시작, 완료, 오류 발생 시 로그를 남기는 데코레이터.
    실행 시간도 함께 기록합니다. (동기 함수용)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"'{operation_name}' 시작...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                logging.info(
                    f"'{operation_name}' 완료 (소요 시간: {time.time() - start_time:.2f}초)"
                )
                return result
            except Exception as e:
                logging.error(f"'{operation_name}' 중 오류 발생: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


# --- 💡 비동기 함수 및 생성기를 위한 새로운 데코레이터 추가 💡 ---
def async_log_operation(operation_name):
    """
    비동기 함수 및 비동기 생성기의 실행 시작, 완료, 오류 발생 시 로그를 남기는 데코레이터.
    스트리밍(yield)을 지원하면서 전체 실행 시간을 정확히 측정합니다.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logging.info(f"'{operation_name}' 시작...")
            start_time = time.time()
            try:
                # 비동기 생성기를 호출하고, 그 결과를 스트리밍합니다.
                async for chunk in func(*args, **kwargs):
                    yield chunk
            except Exception as e:
                logging.error(f"'{operation_name}' 중 오류 발생: {e}", exc_info=True)
                raise
            finally:
                # 스트리밍이 모두 끝난 후에 완료 로그를 남깁니다.
                logging.info(
                    f"'{operation_name}' 완료 (소요 시간: {time.time() - start_time:.2f}초)"
                )

        return wrapper

    return decorator