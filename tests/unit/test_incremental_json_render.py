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
