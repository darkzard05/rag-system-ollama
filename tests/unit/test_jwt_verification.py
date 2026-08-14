"""F12: SimpleJWT verification guarantee locks.

SimpleJWT's real API is ``create_token(user_id, expires_in)`` /
``verify_token(token)`` (there is no generic ``encode(payload)``), and the
payload uses ``user_id`` rather than ``sub``. These tests pin the security
guarantees of ``verify_token``: signature verified via HMAC
``compare_digest`` over ``header.payload`` BEFORE payload decode, then the
``exp`` check, with every failure path returning ``None``.
"""

import base64
import json
import time

from security.auth_system import SimpleJWT


def _new_jwt(tmp_path) -> SimpleJWT:
    return SimpleJWT(secret_file=str(tmp_path / "jwt_sec"))


def test_tampered_payload_rejected(tmp_path):
    jwt = _new_jwt(tmp_path)
    token = jwt.create_token("admin")

    header_seg, payload_seg, sig_seg = token.split(".")
    tampered_payload = payload_seg[:-1] + ("A" if payload_seg[-1] != "A" else "B")
    tampered = f"{header_seg}.{tampered_payload}.{sig_seg}"

    assert jwt.verify_token(tampered) is None


def test_alg_claim_tamper_rejected(tmp_path):
    jwt = _new_jwt(tmp_path)
    token = jwt.create_token("admin")

    header_seg, payload_seg, sig_seg = token.split(".")
    header = json.loads(
        base64.urlsafe_b64decode(header_seg + "=" * (-len(header_seg) % 4))
    )
    header["alg"] = "none"
    new_header_seg = (
        base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    )
    tampered = f"{new_header_seg}.{payload_seg}.{sig_seg}"

    assert jwt.verify_token(tampered) is None


def test_expired_token_rejected(tmp_path):
    jwt = _new_jwt(tmp_path)
    token = jwt.create_token("admin", expires_in=-10)

    assert jwt.verify_token(token) is None


def test_valid_token_roundtrip(tmp_path):
    jwt = _new_jwt(tmp_path)
    token = jwt.create_token("admin", expires_in=3600)

    payload = jwt.verify_token(token)

    assert payload is not None
    assert payload["user_id"] == "admin"
    assert isinstance(payload["jti"], str)
    assert payload["jti"]
    assert payload["exp"] > time.time()


def test_jwt_signature_via_crypto_utils(tmp_path):
    """create_token/verify_token must use crypto_utils.hmac_sign (R14)."""
    from security import crypto_utils

    jwt = _new_jwt(tmp_path)
    token = jwt.create_token("admin", expires_in=3600)

    header_seg, payload_seg, sig_seg = token.split(".")
    message = f"{header_seg}.{payload_seg}"
    expected_sig = crypto_utils.hmac_sign(jwt.secret_key, message, "sha256")

    # The stored signature (base64-decoded) is the raw digest bytes of expected_sig.
    from base64 import urlsafe_b64decode

    stored = urlsafe_b64decode(sig_seg + "=" * (-len(sig_seg) % 4))
    assert stored == bytes.fromhex(expected_sig)

    assert jwt.verify_token(token) is not None
