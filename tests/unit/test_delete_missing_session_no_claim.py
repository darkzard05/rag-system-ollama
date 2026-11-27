"""Deleting a missing session must not reserve its id for the caller."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import api.routes_chat as routes_chat
from api._deps import _session_owners


@pytest.fixture(autouse=True)
def _clear_owners():
    _session_owners.clear()
    yield
    _session_owners.clear()


@pytest.mark.asyncio
async def test_delete_missing_session_does_not_claim_it() -> None:
    sid = "missing-delete-session"
    srv = SimpleNamespace(
        SessionManager=SimpleNamespace(
            session_exists=lambda _sid: False,
            delete_session=lambda _sid: False,
        )
    )
    with patch.object(routes_chat, "_app_server_module", return_value=srv):
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.delete_session(sid, user_id="user-a")

    assert exc_info.value.status_code == 404
    assert sid not in _session_owners
