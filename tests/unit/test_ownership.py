"""Unit tests for ownership-registry sweep logic.

Extracted from ``tests/integration/test_ownership_hardening.py``. These assert
the in-memory ownership registries (``_session_owners`` / ``_file_owners`` in
``src/api/api_server.py``) are purged correctly by ``_sweep_stale_owners``.
They call the function directly (no ASGITransport / HTTP client), so they are
true unit tests with no model or network dependency.
"""

import time

import pytest
import src.api.api_server as api


@pytest.fixture(autouse=True)
def _reset_ownership_registry():
    """모듈 레벨 소유권 레지스트리를 테스트 간 격리합니다."""
    yield
    api._session_owners.clear()
    api._file_owners.clear()


def test_stale_session_owner_swept():
    """TTL 을 초과한 세션 소유권 항목은 _sweep_stale_owners 로 제거되어야 합니다."""
    api._session_owners["ghost-session"] = ("user-x", time.time() - 10 * 24 * 3600)
    api._sweep_stale_owners()
    assert "ghost-session" not in api._session_owners


def test_stale_file_owner_swept():
    """디스크에 존재하지 않는 파일 소유권 항목은 _sweep_stale_owners 로 제거되어야 합니다."""
    api._file_owners["b" * 64] = ("user-x", time.time())
    api._sweep_stale_owners()
    assert "b" * 64 not in api._file_owners


def test_second_user_cannot_access_claimed_session():
    """타인이 점유(claim)한 세션에 다른 사용자가 접근하면 403 이어야 합니다 (fail-closed)."""
    from fastapi import HTTPException

    api._session_owners.clear()
    session_id = "claimed-session"
    # user-a 가 최초로 세션 소유권을 점유
    api._require_session_owner(session_id, "user-a")
    # user-b 가 동일 세션에 접근 시도 → 403
    with pytest.raises(HTTPException) as exc_info:
        api._require_session_owner(session_id, "user-b")
    assert exc_info.value.status_code == 403


def test_first_user_claims_unbound_session():
    """미등록 세션은 최초 인증 사용자가 소유권을 점유하며 403 이 발생하지 않습니다."""
    api._session_owners.clear()
    session_id = "fresh-session"
    # 최초 점유는 예외를 발생시키지 않아야 함
    api._require_session_owner(session_id, "user-a")
    assert api._session_owners[session_id][0] == "user-a"
