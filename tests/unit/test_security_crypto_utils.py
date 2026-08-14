"""Unit tests for :mod:`security.crypto_utils`.

Covers:
- HMAC sign/verify round-trip with a caller-supplied secret.
- Tamper detection via constant-time ``hmac.compare_digest``.
- Token entropy/length guarantees from ``generate_token``.
- No shared-secret leakage: two callers using *different* secrets must produce
  *different* signatures for the same message (R14 — per-call secret).
"""

from __future__ import annotations

import re

import pytest

from security.crypto_utils import (
    generate_token,
    hmac_sign,
    hmac_verify,
)

_MSG = "the-quick-brown-fox"
_SECRET_A = "secret-source-a-000000000000000000000000"
_SECRET_B = "secret-source-b-111111111111111111111111"


def test_hmac_sign_verify_roundtrip_str() -> None:
    sig = hmac_sign(_SECRET_A, _MSG)
    assert isinstance(sig, str)
    assert hmac_verify(_SECRET_A, _MSG, sig)


def test_hmac_sign_verify_roundtrip_bytes() -> None:
    raw = _MSG.encode("utf-8")
    sig = hmac_sign(_SECRET_A, raw)
    assert hmac_verify(_SECRET_A, raw, sig)


def test_hmac_verify_detects_tampered_message() -> None:
    sig = hmac_sign(_SECRET_A, _MSG)
    assert not hmac_verify(_SECRET_A, "tampered-message", sig)


def test_hmac_verify_detects_tampered_signature() -> None:
    sig = hmac_sign(_SECRET_A, _MSG)
    bad_sig = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    assert not hmac_verify(_SECRET_A, _MSG, bad_sig)


def test_hmac_verify_rejects_wrong_secret() -> None:
    sig = hmac_sign(_SECRET_A, _MSG)
    assert not hmac_verify(_SECRET_B, _MSG, sig)


def test_hmac_sign_hex_format_and_stable() -> None:
    sig1 = hmac_sign(_SECRET_A, _MSG)
    sig2 = hmac_sign(_SECRET_A, _MSG)
    assert sig1 == sig2
    assert re.fullmatch(r"[0-9a-f]+", sig1) is not None
    # sha256 => 64 hex chars
    assert len(sig1) == 64


def test_different_secrets_produce_different_signatures() -> None:
    """R14: no shared-secret leakage across independent callers."""
    sig_a = hmac_sign(_SECRET_A, _MSG)
    sig_b = hmac_sign(_SECRET_B, _MSG)
    assert sig_a != sig_b


def test_generate_token_length_and_uniqueness() -> None:
    token = generate_token(32)
    # token_urlsafe(32) -> ceil(32*8/6) = 43 chars (after stripping padding)
    assert len(token) == 43
    assert token != generate_token(32)


def test_generate_token_urlsafe_charset() -> None:
    token = generate_token(16)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", token) is not None


def test_generate_token_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_token(0)
    with pytest.raises(ValueError, match="positive"):
        generate_token(-1)


def test_generate_token_custom_size() -> None:
    token = generate_token(8)
    assert len(token) == 11  # ceil(64/6)
