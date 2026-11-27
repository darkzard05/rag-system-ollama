"""A revoked token must not verify after its signature segment is mutated."""

import base64

from security.auth_system import AuthenticationManager, SimpleJWT


def _same_signature_bytes_variants(token: str) -> list[str]:
    """Build token strings whose signature segment decodes to the same bytes."""
    header, payload, signature = token.split(".")

    def padded(part: str) -> str:
        return part + "=" * (-len(part) % 4)

    signature_bytes = base64.b64decode(padded(signature), altchars=b"-_", validate=True)

    variants: list[str] = []
    for urlsafe, standard in (("-", "+"), ("_", "/")):
        if urlsafe in signature:
            variants.append(signature.replace(urlsafe, standard))

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    for char in alphabet:
        candidate = signature[:-1] + char
        if candidate == signature:
            continue
        if (
            base64.b64decode(padded(candidate), altchars=b"-_", validate=True)
            == signature_bytes
        ):
            variants.append(candidate)

    return [f"{header}.{payload}.{variant}" for variant in variants]


def test_jwt_rejects_non_base64_signature_suffix(tmp_path):
    jwt = SimpleJWT(secret_file=str(tmp_path / "jwt_secret"))
    token = jwt.create_token("admin", expires_in=3600)

    assert jwt.verify_token(token) is not None
    assert jwt.verify_token(token + "!!") is None
    assert jwt.verify_token(token + "!!!") is None
    assert jwt.verify_token(token + "=") is None
    assert jwt.verify_token(token + "==") is None


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
    assert auth.verify_token(token + "=") is None


def test_jwt_rejects_noncanonical_signature_encoding(tmp_path):
    jwt = SimpleJWT(secret_file=str(tmp_path / "jwt_secret"))
    token = jwt.create_token("admin", expires_in=3600)
    variants = _same_signature_bytes_variants(token)
    assert variants
    for variant in variants:
        assert jwt.verify_token(variant) is None


def test_revoked_token_cannot_be_resurrected_by_signature_rewrite(tmp_path):
    auth = AuthenticationManager(
        state_file=str(tmp_path / "auth_state.json"),
        secret_file=str(tmp_path / "jwt_secret"),
    )
    assert auth.register_user("u", "user", "pw") is True
    token = auth._jwt.create_token("u", expires_in=3600)

    assert auth.verify_token(token) == "u"
    assert auth.revoke_token(token) is True
    assert auth.verify_token(token) is None
    for variant in _same_signature_bytes_variants(token):
        assert auth.verify_token(variant) is None
