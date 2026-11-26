"""One damaged page must not discard the coordinates of the remaining pages."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from core.document_hydrator import _extract_pages_words_sync


class _Page:
    def __init__(self, words: list | None, error: Exception | None = None):
        self._words = words
        self._error = error

    def get_text(self, *args, **kwargs):
        if self._error:
            raise self._error
        return self._words


def test_page_error_does_not_abort_later_pages() -> None:
    words = [(1.0, 2.0, 3.0, 4.0, "ok")]
    pages = [_Page([], RuntimeError("damaged content stream")), _Page(words)]
    doc = [pages[0], pages[1]]

    @contextmanager
    def _open(_path):
        yield doc

    with patch("common.pdf_utils.open_pdf_document", side_effect=_open):
        result = _extract_pages_words_sync("doc.pdf", [(1, None), (2, None)])

    assert result[0] is None
    assert result[1] == words
