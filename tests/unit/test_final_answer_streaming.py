"""Regression tests for chunk-boundary quote handling in the final_answer scanner.

Covers two defects in structured raw-JSON streaming:

1. A ``"`` sitting at the end of a chunk was always treated as the closing
   quote, truncating the rest of the answer.
2. A closing ``"`` followed by whitespace/newlines before a ``.``/``,``/``}``
   delimiter was not recognized, leaking raw JSON (``citations``/``confidence``)
   into the answer.
"""

from __future__ import annotations

import json

from common.final_answer import FinalAnswerExtractor


def test_chunk_boundary_inner_quote_preserved() -> None:
    """A quote at a chunk boundary is preserved, not treated as close."""
    extractor = FinalAnswerExtractor()

    deltas = [
        extractor.feed('{"final_answer": "이것은 "'),
        extractor.feed('중요한"'),
        extractor.feed(' , "citations": []}'),
    ]
    deltas.append(extractor.finalize())

    assert "".join(deltas) == '이것은 "중요한'
    assert extractor.closed is True


def test_closing_quote_followed_by_whitespace_delimiter() -> None:
    """Whitespace/newline between a closing quote and a delimiter is recognized."""
    extractor = FinalAnswerExtractor()

    deltas = [
        extractor.feed('{"final_answer": "정답"'),
        extractor.feed(
            '\n , "citations": [{"doc_id": "1", "anchor": "a"}], "confidence": 0.9}'
        ),
    ]
    deltas.append(extractor.finalize())
    result = "".join(deltas)

    assert result == "정답"
    assert "citations" not in result
    assert "confidence" not in result


def test_inner_colon_quote_not_false_close() -> None:
    """A value-internal ``"`` before a colon must not close the value."""
    extractor = FinalAnswerExtractor()

    delta = extractor.feed('{"final_answer": "참고 "A": 설명입니다.", "citations": []}')

    assert delta == '참고 "A": 설명입니다.'
    assert extractor.closed is True


def test_escape_sequences_across_chunk_boundaries() -> None:
    """``\\"``, ``\\\\``, ``\\n``, ``\\uXXXX`` decode when split across chunks."""
    answer = 'a"b\\c\nd한'
    blob = json.dumps({"final_answer": answer}, ensure_ascii=True)

    extractor = FinalAnswerExtractor()
    deltas = [extractor.feed(blob[i : i + 3]) for i in range(0, len(blob), 3)]
    deltas.append(extractor.finalize())

    assert "".join(deltas) == answer


def test_pretty_printed_json_repair_and_extract() -> None:
    """Pretty-printed JSON with inner unescaped quotes repairs and extracts."""
    from core.graph._graph_core import _extract_json_string, _repair_json

    blob = '{\n  "final_answer": "그는 "중요"하다고 말했다." ,\n  "citations": []\n}'

    repaired = _repair_json(blob)
    assert repaired is not None
    parsed = json.loads(repaired)
    assert parsed["final_answer"] == '그는 "중요"하다고 말했다.'

    value, _ = _extract_json_string(blob, blob.index('"', blob.index(":")))
    assert value == '그는 "중요"하다고 말했다.'

    padded = '{\n  "final_answer": "줄 끝" \n}'
    padded_value, _ = _extract_json_string(padded, padded.index('"', padded.index(":")))
    assert padded_value == "줄 끝"
