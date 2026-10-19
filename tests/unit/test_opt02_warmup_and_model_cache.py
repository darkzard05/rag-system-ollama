"""OPT-02: 프로세스 전역 프리웜 1회 + Ollama 모델 목록 TTL 캐시 검증."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import src.main as main_module

import core._loaders as loaders


def _reset_models_cache() -> None:
    loaders.clear_available_models_cache()


def test_fetch_available_models_cached_hits_cache() -> None:
    """동일 TTL 내 재조회는 ollama.Client.list()를 다시 호출하지 않는다."""
    _reset_models_cache()
    fake_client = MagicMock()
    fake_client.list.return_value = {"models": [{"model": "m-b"}, {"model": "m-a"}]}

    with patch("ollama.Client", return_value=fake_client) as client_cls:
        first = loaders._fetch_available_models_cached()
        second = loaders._fetch_available_models_cached()

    assert first == ["m-a", "m-b"]
    assert second == ["m-a", "m-b"]
    client_cls.assert_called_once()


def test_fetch_available_models_cached_expires_after_ttl() -> None:
    """TTL 경과 후에는 재조회한다."""
    _reset_models_cache()
    fake_client = MagicMock()
    fake_client.list.return_value = {"models": [{"model": "m-a"}]}

    with patch("ollama.Client", return_value=fake_client) as client_cls:
        loaders._fetch_available_models_cached()
        with patch.object(loaders, "_MODELS_CACHE_TTL_SECONDS", -1.0):
            loaders._fetch_available_models_cached()

    assert client_cls.call_count == 2


def test_fetch_available_models_cached_caches_failure_only_briefly() -> None:
    """조회 실패는 짧은 TTL로만 캐시해 Ollama 기동 후 빠르게 복구된다."""
    _reset_models_cache()
    fake_client = MagicMock()
    fake_client.list.side_effect = [RuntimeError("down"), {"models": [{"model": "m"}]}]

    with patch("ollama.Client", return_value=fake_client):
        assert loaders._fetch_available_models_cached() == []
        # 실패 TTL 이내에는 재호출하지 않는다 (매 요청 5s 타임아웃 방지).
        assert loaders._fetch_available_models_cached() == []
        # 실패 TTL 경과 후에는 재조회해 복구한다.
        with patch.object(loaders, "_MODELS_FAILURE_TTL_SECONDS", -1.0):
            assert loaders._fetch_available_models_cached() == ["m"]


def test_clear_available_models_cache_forces_refetch() -> None:
    """새로고침 경로(clear)는 다음 호출에서 재조회한다."""
    _reset_models_cache()
    fake_client = MagicMock()
    fake_client.list.return_value = {"models": [{"model": "m"}]}

    with patch("ollama.Client", return_value=fake_client) as client_cls:
        loaders._fetch_available_models_cached()
        loaders.clear_available_models_cache()
        loaders._fetch_available_models_cached()

    assert client_cls.call_count == 2


def test_warmup_guard_is_process_global_not_session_scoped() -> None:
    """프리웜 1회 가드는 session_state가 아니라 프로세스 전역 리소스 캐시여야 한다."""
    source = inspect.getsource(main_module._trigger_global_warmup_once)
    assert "_async_worker.submit" in source
    assert "_warmup_submitted" not in source
    assert hasattr(main_module._trigger_global_warmup_once, "clear"), (
        "st.cache_resource 싱글턴이어야 프로세스 단위 1회가 보장된다"
    )
    main_source = inspect.getsource(main_module)
    assert 'st.session_state.get("_warmup_submitted"' not in main_source


def test_warmup_guard_submits_once_across_sessions() -> None:
    """clear 후 두 번 호출해도 submit은 1회만 발생한다."""
    main_module._trigger_global_warmup_once.clear()
    with patch.object(main_module, "_async_worker") as worker:
        worker.submit.side_effect = lambda coro: coro.close()
        main_module._trigger_global_warmup_once()
        main_module._trigger_global_warmup_once()

    assert worker.submit.call_count == 1
    main_module._trigger_global_warmup_once.clear()


def test_model_list_cache_is_process_global_core_ttl() -> None:
    """모델 목록 캐시는 세션 스코프 st.cache_data가 아니라 코어 전역 TTL 캐시."""
    main_source = inspect.getsource(main_module)
    assert (
        "@st.cache_data(ttl=300, show_spinner=False)\ndef _get_available_models"
        not in (main_source)
    )
    assert "clear_available_models_cache" in main_source, (
        "새로고침 경로가 코어 TTL 캐시를 무효화해야 한다"
    )
    assert hasattr(loaders, "_models_cache_lock")
    assert hasattr(loaders, "clear_available_models_cache")


def test_boot_warmup_thread_is_singleton() -> None:
    """모델 목록 부트 웜업 스레드도 rerun마다 재생성되지 않아야 한다."""
    assert hasattr(main_module._start_model_list_warmup, "clear")
    source = inspect.getsource(main_module._start_model_list_warmup)
    assert "threading.Thread" in source
