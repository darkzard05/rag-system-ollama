"""Unit tests for `_extract_final_answer_delta` incremental JSON value rendering."""

import json

from ui.components.streaming import _extract_final_answer_delta


def test_incremental_accumulation_no_dup():
    """Feed the buffer progressively, reusing the returned scan position, and
    assert no duplicated characters and no JSON syntax leaks into the deltas."""
    b1 = '{"final_answer":"He'
    d1, p1 = _extract_final_answer_delta(b1, 0)

    b2 = '{"final_answer":"Hello'
    d2, p2 = _extract_final_answer_delta(b2, p1)

    b3 = '{"final_answer":"Hello"}'
    d3, p3 = _extract_final_answer_delta(b3, p2)

    assert d1 == "He"
    assert d2 == "llo"
    assert d3 == ""
    assert "".join([d1, d2, d3]) == "Hello"
    for d in (d1, d2, d3):
        assert "{" not in d
        assert "}" not in d
        assert ":" not in d
        assert "final_answer" not in d


def test_fence_strip_and_parse():
    """Replicate the completion-swap fence-strip + json.loads path in isolation
    and verify the `final_answer` field is recovered."""
    cleaned = '```json\n{"final_answer":"Hi","reasoning":"x"}\n```'
    # Replicate the 4-line strip logic from the completion-swap path.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
    if cleaned.startswith("json"):
        cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    parsed = json.loads(cleaned)
    assert parsed["final_answer"] == "Hi"


def test_open_value_partial():
    """Mid-value partial buffer emits the partial text; later buffer completes
    the value without re-emitting already-delivered characters."""
    b = '{"final_answer":"Hel'
    d, p = _extract_final_answer_delta(b, 0)
    assert "Hel" in d or d == "Hel"

    b2 = '{"final_answer":"Hello"}'
    d2, p2 = _extract_final_answer_delta(b2, p)
    assert d2 == "lo"
    assert "".join([d, d2]) == "Hello"


def _make_doc(doc_id: str, page: int, content: str) -> "object":
    """Minimal mapping-style document stand-in for citation rendering tests."""

    class _Doc:
        def __init__(self, doc_id: str, page: int, content: str):
            self.page_content = content
            self.metadata = {"doc_id": doc_id, "page": page, "source": "t.pdf"}

    return _Doc(doc_id, page, content)


def test_citations_array_rendered_by_doc_id():
    """A response carrying a `citations[]` array must surface clickable anchors
    resolved by stable `doc_id` (NOT page number).

    Regression guard for P3: the structured citations array must reach the
    rendered HTML as `data-doc-id` anchors pointing at the correct document.
    """
    from common.utils import apply_tooltips_to_response

    known_doc_id = "doc_abc123"
    # Document lives on page 7; an anchor must resolve to doc_abc123, never p1.
    documents = [_make_doc(known_doc_id, 7, "Deep content about topic X.")]

    citations = [
        {
            "doc_id": known_doc_id,
            "text_span": "topic X detail",
            "section": "§3",
            "page": 7,
            "score": 0.91,
        }
    ]

    html_out = apply_tooltips_to_response(
        "The model says topic X applies.",
        documents,
        citations=citations,
    )

    # Anchor present and keyed by stable doc_id.
    assert f'data-doc-id="{known_doc_id}"' in html_out
    # Must NOT fall back to page-1 mis-link (doc_id, not page, is the key).
    assert 'data-doc-id="1"' not in html_out
    # The cited source label is surfaced.
    assert "topic X detail" in html_out
    # Inline [doc:N] fallback path is preserved/independent.
    assert "citation-sources" in html_out


def test_citations_array_ignored_without_documents():
    """When no documents are supplied, citations are not injected (no dead
    anchors pointing at nothing)."""
    from common.utils import apply_tooltips_to_response

    out = apply_tooltips_to_response("Plain answer with no doc context.")
    assert "citation-sources" not in out
