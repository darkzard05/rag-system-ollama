"""Repro: auth state/secret files must be created with 0600, no tmp window."""

import json
import os
import stat
import sys
from unittest.mock import patch

from src.security.auth_system import AuthenticationManager, SimpleJWT


def test_auth_state_secure_permissions(tmp_path):
    state = tmp_path / "auth_state.json"
    secret = tmp_path / ".jwt_secret"
    chmod_calls: list[tuple[str, int]] = []
    real_chmod = os.chmod

    def _spy_chmod(path, mode):
        chmod_calls.append((str(path), mode))
        return real_chmod(path, mode)

    with patch("src.security.auth_system.os.chmod", side_effect=_spy_chmod):
        m = AuthenticationManager(state_file=str(state), secret_file=str(secret))
        assert m.register_user("u1", "user1", "P@ssw0rd!") is True
        api_key = m.create_api_key("u1")
        assert api_key.startswith("sk_")

    payload = json.loads(state.read_text(encoding="utf-8"))
    assert "password_hash" in payload["users"]["u1"]
    assert api_key in payload["api_keys"]

    # Contract: every secure write must request 0600 (code-level, cross-platform).
    modes = [mode for _, mode in chmod_calls]
    assert any(mode == 0o600 for mode in modes), f"no 0600 chmod: {chmod_calls}"
    assert any(str(state) in p for p, _ in chmod_calls), "state file never chmodded"

    if sys.platform != "win32":
        mode = stat.S_IMODE(os.stat(state).st_mode)
        assert mode & 0o077 == 0, f"world/group-readable: {oct(mode)}"
        smode = stat.S_IMODE(os.stat(secret).st_mode)
        assert smode & 0o077 == 0, f"secret readable: {oct(smode)}"


def test_jwt_secret_uses_secure_create(tmp_path, monkeypatch):
    secret = tmp_path / ".jwt_secret"
    opened_modes: list[int] = []
    real_open = os.open

    def _spy_open(path, flags, *args):
        if args:
            opened_modes.append(args[0])
        return real_open(path, flags, *args)

    monkeypatch.setattr("src.security.auth_system.os.open", _spy_open)
    SimpleJWT(secret_file=str(secret))
    assert secret.exists()
    # Secure creation passes 0o600 to os.open (avoids tmp world-readable window).
    assert any(m == 0o600 for m in opened_modes), f"os.open modes: {opened_modes}"
