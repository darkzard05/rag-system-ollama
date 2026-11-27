"""Upload must pin the embedder it actually hands to the build pipeline."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import UploadFile

import api.routes_docs as routes_docs


@pytest.mark.asyncio
async def test_upload_pins_the_selected_session_embedder(tmp_path: Path) -> None:
    embedder = object()
    pin_calls: list[tuple[str | None, object | None]] = []
    build_embedders: list[object] = []

    class FakeResourceManager:
        async def get_embedder_for_session(self, session_id, model_name):
            return embedder

        @asynccontextmanager
        async def use_embedder(self, model_name=None, embedder=None):
            pin_calls.append((model_name, embedder))
            yield embedder

    class FakeRAGSystem:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

        async def build_pipeline(self, *, embedder, **_kwargs):
            build_embedders.append(embedder)
            return "indexed", False

    srv = SimpleNamespace(
        SessionManager=SimpleNamespace(
            init_session=lambda **_kwargs: None,
            set=lambda *_args, **_kwargs: None,
        ),
        RAGSystem=FakeRAGSystem,
        MAX_PDF_SIZE_BYTES=50 * 1024 * 1024,
        PDF_STORAGE_DIR=str(tmp_path),
    )
    upload = UploadFile(
        filename="doc.pdf", file=io.BytesIO(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
    )

    with (
        patch.object(routes_docs, "_app_server_module", return_value=srv),
        patch.object(
            routes_docs, "get_resource_manager", return_value=FakeResourceManager()
        ),
        patch("common.pdf_utils.get_pdf_page_count_from_bytes", return_value=1),
    ):
        await routes_docs.upload_document(
            file=upload,
            session_id="pin-session",
            embedding_model="custom-embed-model",
            user_id="user-a",
        )

    assert build_embedders == [embedder]
    assert pin_calls == [(None, embedder)]
