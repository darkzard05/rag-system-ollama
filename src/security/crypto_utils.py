"""Shared cryptographic helpers.

This module provides lightweight, dependency-free primitives used by the
security subsystems:

- ``hmac_sign`` / ``hmac_verify`` — HMAC sign/verify with a **per-call** secret.
- ``generate_token`` — cryptographically strong random token generation.

Design notes (R11 / R14):
- The HMAC secret is always passed as a *per-call* argument. There is **no**
  module-global secret, because the future callers keep independent secret
  sources (``cache_security`` uses ``SecurityManager.hmac_secret``;
  ``auth_system`` uses ``SimpleJWT.secret_key`` / ``secret_file``). Each caller
  supplies its own secret on every invocation.
- Password hashing via ``hashlib.pbkdf2_hmac`` (see ``auth_system.py``) is
  intentionally **out of scope** here and must stay in ``auth_system``. This
  module covers only HMAC sign/verify and token generation.
"""

from __future__ import annotations

import hmac
import secrets

__all__ = ["hmac_sign", "hmac_verify", "generate_token"]


def hmac_sign(secret: str, msg: str | bytes, algo: str = "sha256") -> str:
    """Return the hex-encoded HMAC digest of ``msg`` signed with ``secret``.

    Args:
        secret: Caller-supplied secret key (never a module global, R14).
        msg: Message to sign, either ``str`` (UTF-8 encoded) or ``bytes``.
        algo: Hash algorithm name passed to ``hashlib.new`` (default ``sha256``).

    Returns:
        Lowercase hex digest string.
    """
    data = msg.encode("utf-8") if isinstance(msg, str) else msg
    digest = hmac.new(secret.encode("utf-8"), data, digestmod=algo)
    return digest.hexdigest()


def hmac_verify(
    secret: str,
    msg: str | bytes,
    sig: str,
    algo: str = "sha256",
) -> bool:
    """Verify ``sig`` against ``msg`` using a constant-time comparison.

    Args:
        secret: Caller-supplied secret key (never a module global, R14).
        msg: Original message, either ``str`` (UTF-8 encoded) or ``bytes``.
        sig: Hex-encoded signature produced by :func:`hmac_sign`.
        algo: Hash algorithm name (default ``sha256``).

    Returns:
        ``True`` if the signature is valid for the given secret and message.
    """
    expected = hmac_sign(secret, msg, algo)
    return hmac.compare_digest(expected, sig)


def generate_token(n: int = 32) -> str:
    """Generate a URL-safe random token.

    Uses :func:`secrets.token_urlsafe` for URL-safe output. ``n`` controls the
    number of random bytes (default 32 → ~43 chars after base64url encoding).

    Args:
        n: Number of random bytes to draw. Must be positive.

    Returns:
        URL-safe random token string.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return secrets.token_urlsafe(n)
