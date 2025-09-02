"""
프로젝트 전반에서 사용되는 유틸리티 함수들을 모아놓은 파일.
"""

import time
import logging
import functools


def log_operation(operation_name):
    """
    함수 실행 시작, 완료, 오류 발생 시 로그를 남기는 데코레이터.
    실행 시간도 함께 기록합니다.
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
