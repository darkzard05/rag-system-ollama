"""A rejected query must not claim ownership of a session it never used."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import api.routes_chat as routes_chat
from api._deps import _session_owners
from api.schemas import QueryRequest


@pytest.fixture(autouse=True)
def _clear_owners():
    _session_owners.clear()
    yield
    _session_owners.clear()


def _fake_server() -> SimpleNamespace:
    return SimpleNamespace(
        SessionManager=SimpleNamespace(
            set_session_id=lambda _sid: None,
            get=lambda *_args, **_kwargs: None,
            init_session=lambda **_kwargs: None,
        )
    )


@pytest.mark.asyncio
async def test_query_without_document_does_not_claim_session() -> None:
    sid = "rejected-query-session"
    with patch.object(routes_chat, "_app_server_module", return_value=_fake_server()):
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.query_rag(
                QueryRequest(query="hello", session_id=sid),
                user_id="user-a",
                _session_ctx=sid,
            )

    assert exc_info.value.status_code == 400
    assert sid not in _session_owners


@pytest.mark.asyncio
async def test_stream_query_without_document_does_not_claim_session() -> None:
    sid = "rejected-stream-session"
    with patch.object(routes_chat, "_app_server_module", return_value=_fake_server()):
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.stream_query_rag(
                QueryRequest(query="hello", session_id=sid),
                MagicMock(),
                user_id="user-a",
                _session_ctx=sid,
            )

    assert exc_info.value.status_code == 400
    assert sid not in _session_owners
