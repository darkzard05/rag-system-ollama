"""
F1: 관리자 자격증명 부트스트랩 보안 테스트.

_bootstrap_credentials 가 비밀번호/API 키를 준비하되, 민감 정보(평문 비밀번호/
API 키 값)를 콘솔(stderr)이나 로그에 절대 출력하지 않음을 검증합니다 (fail-closed
비밀 출력 금지).

주의: _bootstrap_credentials 는 공유 auth_manager 의 TEST_USER 크리덴셜을
덮어쓰므로, 테스트 격리를 위해 전용 유저(id=BOOTSTRAP_USER)로 실행하고 종료 후
원래 admin 크리덴셜(TEST_PASSWORD)을 복원합니다.
"""

import os

import pytest

import src.api.api_server as api

BOOTSTRAP_USER = "bootstrap_test_user"


@pytest.fixture(autouse=True)
def _isolate_bootstrap_user(monkeypatch):
    """부트스트랩이 공유 admin 크리덴셜을 오염시키지 않도록 전용 유저로 실행합니다."""
    monkeypatch.setattr(api, "TEST_USER", BOOTSTRAP_USER)
    yield
    # 다른 테스트가 의존하는 원래 admin 크리덴셜 복원
    api.auth_manager.upsert_admin_credentials(
        "admin", "admin_user", os.getenv("TEST_ADMIN_PASSWORD") or api.TEST_PASSWORD
    )


def test_boot_generates_random_password_when_env_unset(monkeypatch, capsys):
    monkeypatch.delenv("TEST_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    pw, key = api._bootstrap_credentials(api.auth_manager)
    assert pw != "admin"
    # 비밀번호 평문이 stderr 에 출력되지 않아야 함 (보안)
    captured = capsys.readouterr()
    assert pw not in captured.err
    assert api.auth_manager.authenticate(BOOTSTRAP_USER, pw) is not None


def test_boot_uses_env_password(monkeypatch, capsys):
    monkeypatch.setenv("TEST_ADMIN_PASSWORD", "EnvPass123!")
    pw, key = api._bootstrap_credentials(api.auth_manager)
    assert pw == "EnvPass123!"
    assert "EnvPass123!" not in capsys.readouterr().err


def test_boot_secret_never_printed_to_stderr_or_logger(monkeypatch, capsys, caplog):
    monkeypatch.delenv("TEST_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    pw, key = api._bootstrap_credentials(api.auth_manager)
    assert key.startswith("sk_")
    # API 키 평문이 stderr 에 출력되지 않아야 함
    assert key not in capsys.readouterr().err
    # 로그에도 API 키 평문이 포함되지 않아야 함
    assert not any(key in r.message for r in caplog.records)
