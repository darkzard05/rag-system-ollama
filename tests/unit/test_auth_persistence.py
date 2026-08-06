"""
F1: 인증 상태 영속화 (auth persistence) RED 테스트.

AuthenticationManager 가 state_file/secret_file 을 받아 사용자, 세션, API 키,
deny_list 를 디스크에 저장하고 재시작 후에도 토큰 검증이 유지되는지 검증합니다.
"""

from src.security.auth_system import AuthenticationManager


def test_valid_token_survives_restart(tmp_path):
    m1 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    m1.register_user("u1", "user1", "pw1")
    token, _ = m1.authenticate("u1", "pw1")

    m2 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    assert m2.verify_token(token) == "u1"


def test_revoked_token_denied_after_restart(tmp_path):
    m1 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    m1.register_user("u1", "user1", "pw1")
    token, _ = m1.authenticate("u1", "pw1")
    assert m1.revoke_token(token) is True

    m2 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    assert m2.verify_token(token) is None


def test_session_and_user_persisted(tmp_path):
    m1 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    m1.register_user("u1", "user1", "pw1")
    _, session_id = m1.authenticate("u1", "pw1")

    m2 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    assert m2.get_session(session_id) is not None
    assert m2.authenticate("u1", "pw1") is not None


def test_logout_revocation_survives_restart(tmp_path):
    m1 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    m1.register_user("u1", "user1", "pw1")
    token, session_id = m1.authenticate("u1", "pw1")
    m1.logout(session_id)

    m2 = AuthenticationManager(
        state_file=str(tmp_path / "s.json"),
        secret_file=str(tmp_path / ".sec"),
    )
    assert m2.verify_token(token) is None
