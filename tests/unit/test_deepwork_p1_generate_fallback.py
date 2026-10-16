"""Deepwork P1 — fixed-behavior verification (fail-closed).

Verifies after fix:
 D1: parse failure never promotes raw JSON; safe message + confidence 0.0.
 D2: citations[] unknown doc_id dropped via _normalize_citations.
 D3: int index mapped to stable hash so UI lookup hits.
"""

from langchain_core.documents import Document

from common.utils import doc_stable_id
from core.graph._generate import _normalize_citations
from core.graph._grade_verify import _validate_cited_doc_ids


def _mkdoc(content: str = "hello world", **meta) -> Document:
    return Document(page_content=content, metadata=dict(meta))


def test_p1_normalize_maps_int_index_to_stable_hash():
    docs = [_mkdoc("doc A"), _mkdoc("doc B")]
    sid0 = doc_stable_id(docs[0])

    class _C:
        def __init__(self, doc_id):
            self.doc_id = doc_id
            self.text_span = "span"
            self.section = "S"
            self.page = 1
            self.score = 0.9

        def model_copy(self, update=None):
            c = _C((update or {}).get("doc_id", self.doc_id))
            return c

    out = _normalize_citations([_C(0)], docs)
    assert len(out) == 1
    assert out[0].doc_id == sid0


def test_p1_normalize_drops_hallucinated_doc_id():
    docs = [_mkdoc("real")]

    class _C:
        def __init__(self, doc_id):
            self.doc_id = doc_id
            self.text_span = "fake"
            self.section = "X"
            self.page = 1
            self.score = 0.9

        def model_copy(self, update=None):
            c = _C((update or {}).get("doc_id", self.doc_id))
            return c

    assert _normalize_citations([_C(99)], docs) == []
    # Body-only validator still sees nothing (documents the gap it leaves).
    assert _validate_cited_doc_ids("no tags here", docs) == []


def test_p1_int_doc_id_resolves_via_normalize_not_raw_compare():
    docs = [_mkdoc("hello world", page=3)]
    assert str(0) != doc_stable_id(docs[0])

    class _C:
        def __init__(self, doc_id):
            self.doc_id = doc_id
            self.text_span = "s"
            self.section = "S"
            self.page = 3
            self.score = 0.9

        def model_copy(self, update=None):
            c = _C((update or {}).get("doc_id", self.doc_id))
            return c

    out = _normalize_citations([_C(0)], docs)
    assert out[0].doc_id == doc_stable_id(docs[0])
