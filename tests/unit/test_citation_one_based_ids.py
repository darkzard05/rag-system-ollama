"""Small-model schema drift: integer ``doc_id`` values may be 1-based.

The structured-output prompt asks for copied hash tokens, but a 4B model can
fall back to plain indices. When those indices are 1..N they currently shift
every citation one document forward. These tests pin the observable mapping.
"""

from __future__ import annotations

from langchain_core.documents import Document

from core.graph._generate import _normalize_citations

_DOCS = [
    Document(page_content="alpha body", metadata={"doc_id": "aaa111"}),
    Document(page_content="beta body", metadata={"doc_id": "bbb222"}),
    Document(page_content="gamma body", metadata={"doc_id": "ccc333"}),
]


def test_one_based_integer_ids_map_to_matching_documents() -> None:
    out = _normalize_citations(
        [{"doc_id": i, "anchor": "kw"} for i in (1, 2, 3)], _DOCS
    )
    assert [c["doc_id"] for c in out] == ["aaa111", "bbb222", "ccc333"]


def test_zero_based_integer_ids_keep_existing_mapping() -> None:
    out = _normalize_citations(
        [{"doc_id": 0, "anchor": "kw"}, {"doc_id": 2, "anchor": "kw"}], _DOCS
    )
    assert [c["doc_id"] for c in out] == ["aaa111", "ccc333"]


def test_partial_one_based_ids_map_to_matching_documents() -> None:
    """A model citing only the first two of five docs still uses 1..N indexing."""
    docs = [
        Document(page_content=f"body {i}", metadata={"doc_id": f"doc{i}"})
        for i in range(5)
    ]
    out = _normalize_citations(
        [{"doc_id": 1, "anchor": "a"}, {"doc_id": 2, "anchor": "b"}], docs
    )
    assert [c["doc_id"] for c in out] == ["doc0", "doc1"]
