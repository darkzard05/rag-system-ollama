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


# ---------------------------------------------------------------------------
# DEFECT-1: streaming state-machine scanner tests
# ---------------------------------------------------------------------------


def test_escape_decode_newline():
    """Escape \\n inside value → decoded newline in delta, not literal \\n."""
    buf = '{"final_answer":"Hello\\nWorld"}'
    d, p = _extract_final_answer_delta(buf, 0)
    assert "Hello\nWorld" in d
    assert "\\n" not in d


def test_escape_decode_quote_and_backslash():
    """Escape \\\" and \\\\ → decoded to literal quote and backslash."""
    buf = r'{"final_answer":"He said \"hi\" and \\"}'
    d, p = _extract_final_answer_delta(buf, 0)
    assert d == 'He said "hi" and \\'


def test_unescaped_inner_quote_full_value():
    """Unescaped inner quote does NOT truncate the value."""
    buf = '{"final_answer":"The answer is "important" and complete"}'
    d, _ = _extract_final_answer_delta(buf, 0)
    assert d == 'The answer is "important" and complete'


def test_delimiter_aware_close():
    """Value closes at delimiter, subsequent call returns empty delta."""
    buf = '{"final_answer":"done","reasoning":"x"}'
    d, p = _extract_final_answer_delta(buf, 0)
    assert d == "done"
    d2, p2 = _extract_final_answer_delta(buf, p)
    assert d2 == ""
    assert p2 == p


def test_chunk_boundary_escape_crosses_chunks():
    """Trailing backslash at chunk boundary is deferred, decoded on next call."""
    c1 = '{"final_answer":"Hello\\'
    c2 = 'nWorld"}'

    d1, p1 = _extract_final_answer_delta(c1, 0)
    assert d1 == "Hello"

    full = c1 + c2
    d2, p2 = _extract_final_answer_delta(full, p1)
    assert d2.startswith("\n")
    assert "World" in d2


def test_chunk_boundary_trailing_backslash():
    """Deterministic: trailing backslash suppressed, decoded on next call."""
    c1 = '{"final_answer":"test\\'
    c2 = 'n"}'

    d1, p1 = _extract_final_answer_delta(c1, 0)
    assert d1 == "test"

    full = c1 + c2
    d2, p2 = _extract_final_answer_delta(full, p1)
    assert d2 == "\n"


def test_key_position_cached():
    """Key search performed at most once when _key_pos is reused."""
    find_count = 0

    class CountingStr(str):
        def find(self, sub: str, start: int = 0) -> int:  # type: ignore[override]
            nonlocal find_count
            if sub == '"final_answer"':
                find_count += 1
            return super().find(sub, start)

    buffers = [
        '{"final_answer":"A',
        '{"final_answer":"AB',
        '{"final_answer":"ABC',
        '{"final_answer":"ABCD',
        '{"final_answer":"ABCDE"}',
    ]
    key_pos: list[int] = [-1]
    pos = 0
    for buf in buffers:
        _d, pos = _extract_final_answer_delta(CountingStr(buf), pos, key_pos)

    assert key_pos[0] >= 0
    assert find_count == 1

    find_count = 0
    pos = 0
    for buf in buffers:
        _d, pos = _extract_final_answer_delta(CountingStr(buf), pos)

    assert find_count == 5


def test_value_closed_early_by_delimiter():
    """Value closed by delimiter → returns value, no crash."""
    buf = '{"final_answer":"hi","other":"x"}'
    d, p = _extract_final_answer_delta(buf, 0)
    assert d == "hi"


def test_key_not_yet_arrived():
    """Key not present → returns empty, _key_pos stays -1."""
    buf = '{"reasoning":"x"}'
    key_pos: list[int] = [-1]
    d, p = _extract_final_answer_delta(buf, 0, key_pos)
    assert d == ""
    assert key_pos[0] == -1
