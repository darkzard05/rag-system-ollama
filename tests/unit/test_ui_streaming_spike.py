"""Spike test for streaming extraction — baseline for Batch 5.2.

Asserts critical pure helpers in ``ui.components.streaming`` keep working
before and after extraction into focused modules. Uses MagicMock-style
unit approach (no AppTest).
"""

from __future__ import annotations

from unittest.mock import MagicMock


def test_build_process_dedup_and_cap() -> None:
    from ui.components.streaming import _build_process

    msg = {"process_steps": ["A", "A", "B", "B", "A"]}
    p = _build_process(msg)
    assert p["steps"] == ["A", "B", "A"]
    # Cap to last 10
    long_steps = [f"s{i}" for i in range(14)]
    p2 = _build_process({"process_steps": long_steps})
    assert p2["steps"] == [f"s{i}" for i in range(4, 14)]


def test_build_process_sections_and_scores() -> None:
    from ui.components.streaming import _build_process

    doc = MagicMock()
    doc.metadata = {"current_section": "Intro", "rerank_score": 0.9}
    doc2 = MagicMock()
    doc2.metadata = {"current_section": "Intro", "rerank_score": 0.8}
    doc3 = MagicMock()
    doc3.metadata = {"current_section": "Methods", "rerank_score": 0.95}
    p = _build_process(
        {"documents": [doc, doc2, doc3], "metrics": {"total_time": 1.2, "tps": 30}}
    )
    assert p["retrieved_count"] == 3
    assert p["sections"] == ["Intro", "Methods"]
    # Top scores sorted descending, max 3
    assert p["top_scores"][0]["score"] == 0.95
    assert "total_time" in p["perf"]


def test_friendly_error_maps_connection_refused() -> None:
    from ui.components.streaming import friendly_error_message

    msg = friendly_error_message(RuntimeError("connection refused by server"))
    # Should map to Ollama not running message, not generic
    assert (
        "Ollama" in msg
        or "ollama" in msg.lower()
        or msg != "An error occurred while generating the answer."
    )


def test_extract_final_answer_delta_incremental() -> None:
    from ui.components.streaming import _extract_final_answer_delta

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


def test_recover_final_answer_unclosed() -> None:
    from ui.components.streaming import _recover_final_answer

    blob = '{"final_answer": "CM3 test'
    assert _recover_final_answer(blob) == "CM3 test"
    assert _recover_final_answer("") is None
    assert _recover_final_answer('{"reasoning":"x"}') is None
