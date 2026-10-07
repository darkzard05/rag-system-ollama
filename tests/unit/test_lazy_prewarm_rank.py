"""Task 4 — lazy citation-rank annotation prewarm (single PDF open).

Given multi-doc fixture sharing one PDF / When extract runs with citations
/ Then a single open per distinct file serves all ranked pages, only cited
pages are computed, and rank order is respected. Viewer single-page invariant
(never compute never-displayed pages eagerly) is preserved by construction:
uncited pages are never searched.
"""

import contextlib
from unittest.mock import patch

from common.utils import extract_annotations_from_docs


def _content_for(page: int) -> str:
    return (
        "The retrieval pipeline extracts precise highlight coordinates "
        f"for citations on page {page} with additional context words here."
    )


class _FakeQuad:
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class _FakePage:
    def __init__(
        self, page_no: int, search_log: list[int], quads: list[_FakeQuad]
    ) -> None:
        self._page_no = page_no
        self._search_log = search_log
        self._quads = quads

    def get_textpage(self) -> object:
        return object()

    def search_for(self, part: str, textpage: object = None) -> list[_FakeQuad]:
        assert len(part) >= 12
        self._search_log.append(self._page_no)
        return list(self._quads)


class _FakePdf:
    def __init__(self, search_log: list[int]) -> None:
        self._search_log = search_log

    def __getitem__(self, idx: int) -> _FakePage:
        page_no = idx + 1
        return _FakePage(
            page_no,
            self._search_log,
            [_FakeQuad(10.0, 20.0, 60.0, 30.0)],
        )


class _OpenSpy:
    """open_pdf_document replacement counting opens per file."""

    def __init__(self, search_log: list[int]) -> None:
        self.opens = 0
        self.opens_by_path: dict[str, int] = {}
        self._search_log = search_log

    @contextlib.contextmanager
    def open(self, path: str):  # type: ignore[no-untyped-def]
        self.opens += 1
        self.opens_by_path[path] = self.opens_by_path.get(path, 0) + 1
        yield _FakePdf(self._search_log)


def _doc(pdf_path: str, page: int) -> dict:
    return {
        "page_content": _content_for(page),
        "metadata": {
            "file_path": pdf_path,
            "page": page,
            "doc_id": f"doc-{page}",
        },
    }


def test_single_open_serves_same_pdf(tmp_path) -> None:
    """Given 3 chunks from one PDF / When extracted / Then 1 open total."""
    pdf = tmp_path / "shared.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    docs = [_doc(str(pdf), p) for p in (1, 2, 3)]
    search_log: list[int] = []
    spy = _OpenSpy(search_log)
    with patch("common.utils.open_pdf_document", spy.open):
        annots = extract_annotations_from_docs(docs)
    assert spy.opens == 1, f"expected single open, got {spy.opens}"
    assert len(annots) > 0
    assert sorted({a["page"] for a in annots}) == [1, 2, 3]


def test_only_cited_pages_in_rank_order(tmp_path) -> None:
    """Given 4 pages but citations rank [3, 1] / When extracted with
    citations / Then only pages 3 and 1 are searched, in rank order,
    with a single open."""
    pdf = tmp_path / "ranked.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    docs = [_doc(str(pdf), p) for p in (1, 2, 3, 4)]
    citations = [
        {"doc_id": "doc-3", "page": 3},
        {"doc_id": "doc-1", "page": 1},
    ]
    search_log: list[int] = []
    spy = _OpenSpy(search_log)
    with patch("common.utils.open_pdf_document", spy.open):
        annots = extract_annotations_from_docs(docs, citations)
    assert spy.opens == 1, f"expected single open, got {spy.opens}"
    searched_pages = sorted(set(search_log))
    assert searched_pages == [1, 3], f"uncited pages computed: {search_log}"
    first_seen: list[int] = []
    for p in search_log:
        if p not in first_seen:
            first_seen.append(p)
    assert first_seen == [3, 1], f"rank order violated: {first_seen}"
    assert sorted({a["page"] for a in annots}) == [1, 3]


def test_distinct_files_open_once_each(tmp_path) -> None:
    """Given docs from 2 PDFs / When extracted / Then 1 open per file."""
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 fake")
    pdf_b.write_bytes(b"%PDF-1.4 fake")
    docs = [
        {
            "page_content": _content_for(1),
            "metadata": {
                "file_path": str(pdf_a),
                "page": 1,
                "doc_id": "a-1",
            },
        },
        {
            "page_content": _content_for(2),
            "metadata": {
                "file_path": str(pdf_b),
                "page": 1,
                "doc_id": "b-1",
            },
        },
    ]
    search_log: list[int] = []
    spy = _OpenSpy(search_log)
    with patch("common.utils.open_pdf_document", spy.open):
        annots = extract_annotations_from_docs(docs)
    assert spy.opens == 2, f"expected 1 open/file, got {spy.opens}"
    assert spy.opens_by_path[str(pdf_a)] == 1
    assert spy.opens_by_path[str(pdf_b)] == 1
    assert len(annots) == 2


def test_no_top_level_fitz_import() -> None:
    """Given common.utils module / When inspected / Then fitz stays lazy."""
    import pathlib

    src = pathlib.Path("src/common/utils.py").read_text(encoding="utf-8")
    top_imports = [
        line
        for line in src.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    assert not any("fitz" in line or "pymupdf" in line for line in top_imports)
