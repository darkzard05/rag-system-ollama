"""
F1: 관리자 자격증명 부트스트랩 RED 테스트.

_bootstrap_credentials 가 비밀번호/API 키를 준비하고, 민감 정보를
파일 로거가 아닌 콘솔(stderr)에만 1회 출력하는지 검증합니다.
"""

import src.api.api_server as api


def test_boot_generates_random_password_when_env_unset(monkeypatch, capsys):
    monkeypatch.delenv("TEST_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    pw, key = api._bootstrap_credentials(api.auth_manager)
    assert pw != "admin"
    captured = capsys.readouterr()
    assert pw in captured.err
    assert api.auth_manager.authenticate(api.TEST_USER, pw) is not None


def test_boot_uses_env_password(monkeypatch, capsys):
    monkeypatch.setenv("TEST_ADMIN_PASSWORD", "EnvPass123!")
    pw, key = api._bootstrap_credentials(api.auth_manager)
    assert pw == "EnvPass123!"
    assert "EnvPass123!" not in capsys.readouterr().err


def test_boot_api_key_printed_to_console_not_logger(monkeypatch, capsys, caplog):
    monkeypatch.delenv("TEST_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    pw, key = api._bootstrap_credentials(api.auth_manager)
    assert key.startswith("sk_")
    assert key in capsys.readouterr().err
    assert not any(key in r.message for r in caplog.records)
