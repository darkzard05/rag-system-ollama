"""단위 테스트 스위트 누적 지연(PBKDF2/재시도 슬립/Ollama 소켓 타임아웃) 회귀 테스트."""

import time

import pytest

from common.config import is_test_env
from common.resilience import retry_with_backoff
from core.model_loader import _fetch_available_models_cached
from security.auth_system import PasswordHasher


def test_pytest_environment_auto_detection(monkeypatch):
    """PYTEST_CURRENT_TEST 감지를 통한 무설정 테스트 환경 인식 검증"""
    monkeypatch.delenv("IS_CI_TEST", raising=False)
    monkeypatch.delenv("IS_UNIT_TEST", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_run")
    assert is_test_env() is True


def test_password_hasher_fast_iteration_in_test():
    """테스트 환경에서 PasswordHasher가 고속 iteration(10회)을 사용하여 5ms 이내에 완료되는지 검증"""
    t0 = time.perf_counter()
    pwd_hash, salt = PasswordHasher.hash_password("test_password")
    elapsed = time.perf_counter() - t0

    assert PasswordHasher.verify_password("test_password", pwd_hash, salt) is True
    # 100,000회는 통상 >20ms 소요. 10회는 <2ms 소요.
    assert elapsed < 0.005, f"PBKDF2 hashing took too long: {elapsed:.4f}s"


def test_fetch_available_models_no_socket_timeout_in_test():
    """테스트 환경에서 Ollama 데몬 부재 시 5초 타임아웃 없이 즉각 반환되는지 검증"""
    t0 = time.perf_counter()
    models = _fetch_available_models_cached(non_blocking_fallback=False)
    elapsed = time.perf_counter() - t0

    assert len(models) > 0
    assert elapsed < 0.05, f"Ollama model fetch blocked for {elapsed:.4f}s"


def test_retry_with_backoff_fast_in_test():
    """테스트 환경에서 retry_with_backoff의 백오프 딜레이가 0.001s로 자동 축소되는지 검증"""
    calls = 0

    def failing_fn():
        nonlocal calls
        calls += 1
        raise ValueError("transient error")

    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="transient error"):
        retry_with_backoff(failing_fn, max_retries=3)
    elapsed = time.perf_counter() - t0

    assert calls == 3
    # 기존: 1.0s + 2.0s = 3.0s 이상 소요. 수정 후: < 0.05s 소요.
    assert elapsed < 0.05, f"Retry loop took too long: {elapsed:.4f}s"
