"""A revoked token must not verify after its signature segment is mutated."""

from security.auth_system import AuthenticationManager, SimpleJWT


def test_jwt_rejects_non_base64_signature_suffix(tmp_path):
    jwt = SimpleJWT(secret_file=str(tmp_path / "jwt_secret"))
    token = jwt.create_token("admin", expires_in=3600)

    assert jwt.verify_token(token) is not None
    assert jwt.verify_token(token + "!!") is None
    assert jwt.verify_token(token + "!!!") is None


def test_revoked_token_cannot_be_resurrected(tmp_path):
    auth = AuthenticationManager(
        state_file=str(tmp_path / "auth_state.json"),
        secret_file=str(tmp_path / "jwt_secret"),
    )
    assert auth.register_user("u", "user", "pw") is True
    token = auth._jwt.create_token("u", expires_in=3600)

    assert auth.verify_token(token) == "u"
    assert auth.revoke_token(token) is True
    assert auth.verify_token(token) is None
    assert auth.verify_token(token + "!!") is None
