"""Logout with an API key must revoke the session JWT after a restart."""

from __future__ import annotations

from pathlib import Path

from src.security.auth_system import AuthenticationManager


def test_api_key_logout_revokes_persisted_session_token(tmp_path: Path) -> None:
    state_file = str(tmp_path / "s.json")
    secret_file = str(tmp_path / ".sec")
    m1 = AuthenticationManager(state_file=state_file, secret_file=secret_file)
    m1.register_user("u1", "user1", "pw1")
    jwt, session_id = m1.authenticate("u1", "pw1")
    api_key = m1.create_api_key("u1", expires_in=3600)
    assert jwt is not None
    assert session_id is not None

    m2 = AuthenticationManager(state_file=state_file, secret_file=secret_file)
    assert m2.revoke_token(api_key) is True
    assert m2.logout(session_id) is True

    assert m2.verify_token(jwt) is None
