"""Unit tests for API upload PDF validation (P1-7).

The API upload path previously checked only the ``.pdf`` filename suffix,
so non-PDF bytes could be persisted to storage and bound in the ownership
registry. ``_ensure_valid_pdf`` mirrors the UI gate (magic byte + fitz
structure check) and rejects junk with 400 before any persist/bind.
"""

import pytest
from fastapi import HTTPException

from api.routes_docs import _ensure_valid_pdf

_FAKE_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"


def test_valid_pdf_bytes_accepted() -> None:
    _ensure_valid_pdf(_FAKE_PDF, "test.pdf")


def test_non_pdf_bytes_rejected() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _ensure_valid_pdf(b"This is not a PDF at all.", "evil.pdf")
    assert exc_info.value.status_code == 400


def test_empty_bytes_rejected() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _ensure_valid_pdf(b"", "empty.pdf")
    assert exc_info.value.status_code == 400


def test_pdf_magic_with_leading_whitespace_accepted() -> None:
    _ensure_valid_pdf(b"\n%PDF-1.4\n%%EOF", "ws.pdf")
