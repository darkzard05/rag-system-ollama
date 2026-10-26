import time
import pytest
from core.session import SessionManager
from core._loaders import _fetch_available_models_cached
import core._loaders as loaders_mod


def test_new_session_initializes_with_dirty_keys():
    """RED: 신규 세션 생성 시 기본 상태가 UI로 미러링될 수 있도록 _dirty_keys가 채워져야 한다."""
    sid = "test_cold_start_session"
    SessionManager.reset()
    SessionManager.init_session(sid)

    assert SessionManager.has_pending_ui_sync(sid) is True, (
        "New session must mark initial state dirty so UI receives baseline SSoT values on first pass"
    )


def test_non_blocking_model_cache_instant_return(monkeypatch):
    """RED: 최초 모델 캐시 부재 상태에서 non_blocking_fallback=True 호출 시 지연 없이 즉시 반환해야 한다."""
    monkeypatch.setattr(loaders_mod, "_models_cache", None)

    t0 = time.perf_counter()
    models = _fetch_available_models_cached(non_blocking_fallback=True)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.01, f"Initial non-blocking fetch took too long: {elapsed:.4f}s"
    assert len(models) > 0
