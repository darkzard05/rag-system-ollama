"""Unit tests for PartialCitationScanner (Phase 0 incremental citations)."""

from __future__ import annotations

from core.graph._partial_citations import PartialCitationScanner

_CITE = '{"doc_id": 1, "text_span": "hello", "section": "s", "page": 1, "score": 0.9}'


def test_chunk_split_mid_object() -> None:
    scanner = PartialCitationScanner()
    blob = '{"final_answer": "x", "citations": [' + _CITE + "]}"
    cut = blob.index('"text_span"')
    assert scanner.feed(blob[:cut]) == []
    assert scanner.dead is False
    out = scanner.feed(blob)
    assert len(out) == 1
    assert out[0][0] == 0
    assert out[0][1]["doc_id"] == 1


def test_escapes_in_strings() -> None:
    scanner = PartialCitationScanner()
    cite = (
        '{"doc_id": "ab\\\\cd", "text_span": "say \\"hi\\" \\u0041", '
        '"section": "s", "page": 2, "score": 0.5}'
    )
    out = scanner.feed('{"citations": [' + cite + "]}")
    assert len(out) == 1
    assert out[0][1]["text_span"] == 'say "hi" A'
    assert out[0][1]["doc_id"] == "ab\\cd"


def test_malformed_object_stays_buffered() -> None:
    scanner = PartialCitationScanner()
    bad = '{"doc_id": 1,}'
    good = '{"doc_id": 2, "text_span": "t", "section": "", "page": 0, "score": 0.0}'
    out = scanner.feed('{"citations": [' + bad + ", " + good + "]}")
    assert len(out) == 1
    assert out[0][1]["doc_id"] == 2
    assert scanner.emitted == 1
    assert scanner.dead is False


def test_unclosed_array_no_harvest_no_dead() -> None:
    scanner = PartialCitationScanner()
    out = scanner.feed('{"citations": [{"doc_id": 1')
    assert out == []
    assert scanner.dead is False
    assert scanner.emitted == 0


def test_citations_before_final_answer_order() -> None:
    scanner = PartialCitationScanner()
    blob = '{"citations": [' + _CITE + '], "final_answer": "done"}'
    out = scanner.feed(blob)
    assert len(out) == 1
    assert out[0][1]["doc_id"] == 1


def test_attempt_reset_fresh_scanner() -> None:
    first = PartialCitationScanner()
    first.feed('{"citations": [' + _CITE + "]}")
    assert first.emitted == 1
    second = PartialCitationScanner()
    assert second.emitted == 0
    assert second.dead is False
    out = second.feed('{"citations": [' + _CITE + "]}")
    assert len(out) == 1
    assert out[0][0] == 0


def test_64kb_suicide_zero_harvest() -> None:
    scanner = PartialCitationScanner()
    blob = '{"citations": [' + (" " * (64 * 1024 + 16))
    assert scanner.feed(blob) == []
    assert scanner.dead is True
    assert scanner.feed(blob + _CITE + "]}") == []
    assert scanner.emitted == 0


def test_marker_split_across_chunks_recovers() -> None:
    blob = '{"final_answer": "x", "citations": [' + _CITE + "]}"
    for step in (1, 7, 13):
        scanner = PartialCitationScanner()
        acc = ""
        harvested = 0
        for i in range(0, len(blob), step):
            acc += blob[i : i + step]
            harvested += len(scanner.feed(acc))
        assert harvested == 1, f"step={step}"
        assert scanner.emitted == 1, f"step={step}"
