import inspect
import logging
from unittest.mock import patch

import pytest

from cache.coord_cache import CoordCacheManager


def test_stop_owner_loop_emits_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="cache.coord_cache")
    manager = CoordCacheManager()
    manager._ensure_owner_loop()
    assert manager._owner_loop is not None
    assert manager._owner_thread is not None
    manager._stop_owner_loop()
    assert any("owner 루프" in record.message for record in caplog.records), (
        "_stop_owner_loop가 stop-path 로그를 발화하지 않음"
    )


def test_atexit_registers_coord_cache_close() -> None:
    import atexit

    captured: list = []
    original_register = atexit.register

    def tracking_register(func, *args, **kwargs):
        captured.append(func)
        return original_register(func, *args, **kwargs)

    with patch("atexit.register", side_effect=tracking_register):
        import src.main

        # reload는 모듈 함수 객체를 복제해 타 테스트의 identity 단언을 깨므로 금지.
        # 이미 import된 모듈 객체의 캐시만 비우고 직접 호출한다 (순서 무관).
        src.main._register_cleanup_handlers.clear()
        src.main._register_cleanup_handlers()
    found = False
    for func in captured:
        try:
            source = inspect.getsource(func)
            if "coord_cache" in source or "close" in source:
                found = True
                break
        except (OSError, TypeError):
            pass
    assert found, "atexit에 coord_cache 정리 함수가 등록되지 않음"
