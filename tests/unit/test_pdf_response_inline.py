"""The PDF endpoint must serve files inline so #page=N navigation works."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import api.routes_docs as routes_docs


@pytest.mark.asyncio
async def test_served_pdf_uses_inline_disposition(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with (
        patch.object(routes_docs, "_require_file_owner"),
        patch.object(routes_docs, "_resolve_pdf_path", return_value=pdf_path),
    ):
        response = await routes_docs.serve_pdf("a" * 64, user_id="user-a")

    disposition = response.headers["content-disposition"]
    assert disposition.startswith("inline")
