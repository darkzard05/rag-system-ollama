"""A rejected upload must not claim ownership of a session it never used."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile

import api.routes_docs as routes_docs
from api._deps import _session_owners


@pytest.fixture(autouse=True)
def _clear_owners():
    _session_owners.clear()
    yield
    _session_owners.clear()


@pytest.mark.asyncio
async def test_invalid_pdf_upload_does_not_claim_session() -> None:
    sid = "rejected-upload-session"
    init_calls: list[str] = []
    srv = SimpleNamespace(
        SessionManager=SimpleNamespace(
            init_session=lambda **kwargs: init_calls.append(kwargs["session_id"])
        ),
        MAX_PDF_SIZE_BYTES=50 * 1024 * 1024,
    )
    upload = UploadFile(filename="bad.pdf", file=io.BytesIO(b"garbage"))

    with (
        patch.object(routes_docs, "_app_server_module", return_value=srv),
        pytest.raises(HTTPException) as exc_info,
    ):
        await routes_docs.upload_document(
            file=upload,
            session_id=sid,
            embedding_model=None,
            user_id="user-a",
        )

    assert exc_info.value.status_code == 400
    assert sid not in _session_owners
    assert init_calls == []
