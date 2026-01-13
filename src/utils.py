"""
프로젝트 전반에서 사용되는 유틸리티 함수들을 모아놓은 파일.
Utils Rebuild: 복잡한 데코레이터 제거 및 비동기 헬퍼 단순화.
"""

import logging
import asyncio
import nest_asyncio
import re
import time
import functools

logger = logging.getLogger(__name__)
_RE_WHITESPACE = re.compile(r'\s+')

def preprocess_text(text: str) -> str:
    """텍스트 정제: 널 문자 및 연속 공백 제거"""
    if not text:
        return ""
    text = text.replace("\x00", "")
    return _RE_WHITESPACE.sub(' ', text).strip()


def sync_run(coro):
    """
    Streamlit(동기 환경)에서 비동기 코루틴을 안전하게 실행하기 위한 헬퍼.
    nest_asyncio를 적용하여 이벤트 루프 중첩 허용.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    
    return asyncio.run(coro)


def log_operation(operation_name):
    """
    단순 동기 함수용 로깅 데코레이터.
    GraphBuilder의 Node 함수에는 사용하지 마세요! (config 전달 문제 발생 가능)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"[작업 시작] '{operation_name}'")
            start = time.time()
            try:
                res = func(*args, **kwargs)
                dur = time.time() - start
                logger.info(f"[작업 완료] '{operation_name}' ({dur:.2f}초)")
                return res
            except Exception as e:
                logger.error(f"[작업 실패] '{operation_name}': {e}")
                raise
        return wrapper
    return decorator
