"""Phase 1: 증분 인용 reconciliation 단위 테스트.

``merge_citations_add`` 의 순수 병합 규칙(index upsert / arr_len 절단 /
doc_id int·str 정규화 dedupe)과 ``_content_generator`` 수준의
final-replace(authoritative) 동작을 검증한다.
"""

from __future__ import annotations

from unittest.mock import patch

from api.stream_pipeline import StreamChunk
from core.session import SessionManager
from ui.components.streaming_core import _content_generator, merge_citations_add

_SID = "recon-sid"
_MSG = "recon-msg"


def _cit(
    doc_id: object, page: int = 1, section: str = "s", text_span: str = "t"
) -> dict:
    return {"doc_id": doc_id, "page": page, "section": section, "text_span": text_span}


def test_index_upsert_replaces_in_place() -> None:
    incr = [_cit("a"), _cit("b")]
    merged = merge_citations_add(incr, items=[_cit("c")], base_index=1, arr_len=2)
    assert [c["doc_id"] for c in merged] == ["a", "c"]


def test_index_upsert_appends_beyond_length() -> None:
    incr = [_cit("a")]
    merged = merge_citations_add(
        incr, items=[_cit("b"), _cit("c")], base_index=1, arr_len=3
    )
    assert [c["doc_id"] for c in merged] == ["a", "b", "c"]


def test_truncation_on_shrink() -> None:
    incr = [_cit("a"), _cit("b"), _cit("c")]
    merged = merge_citations_add(incr, items=[], base_index=0, arr_len=2)
    assert [c["doc_id"] for c in merged] == ["a", "b"]


def test_doc_id_int_str_normalization_dedupes() -> None:
    incr = [_cit(5, page=1, section="s", text_span="t")]
    merged = merge_citations_add(
        incr,
        items=[_cit("5", page=1, section="s", text_span="t")],
        base_index=1,
        arr_len=2,
    )
    assert len(merged) == 1


def test_dedupe_secondary_key_keeps_first() -> None:
    incr = [_cit("a", page=1, section="s", text_span="hello world")]
    merged = merge_citations_add(
        incr,
        items=[_cit("a", page=1, section="s", text_span="hello  world")],
        base_index=1,
        arr_len=2,
    )
    # text_span 해시는 원문 기준이므로 공백이 다르면 별개로 본다.
    assert len(merged) == 2


def test_final_replace_is_authoritative() -> None:
    """최종 citations 이벤트는 증분 목록을 통째로 대체한다."""
    SessionManager.reset_all_state(_SID)
    chunks = [
        StreamChunk(
            citations_add={
                "items": [_cit("a"), _cit("b")],
                "base_index": 0,
                "arr_len": 2,
                "attempt": 0,
            }
        ),
        StreamChunk(citations=[_cit("z")]),
    ]
    with patch("ui.components.streaming_core.stream_chunks", return_value=iter(chunks)):
        list(_content_generator("q", "m", _SID, _MSG))
    final = SessionManager.get("stream_active_aux", None, _SID)
    assert final is not None
    assert final["citations"] == [_cit("z")]


def test_attempt_change_resets_incremental_list() -> None:
    """새 attempt가 오면 증분 목록을 리셋한다."""
    SessionManager.reset_all_state(_SID)
    chunks = [
        StreamChunk(
            citations_add={
                "items": [_cit("a")],
                "base_index": 0,
                "arr_len": 1,
                "attempt": 0,
            }
        ),
        StreamChunk(
            citations_add={
                "items": [_cit("x")],
                "base_index": 0,
                "arr_len": 1,
                "attempt": 1,
            }
        ),
    ]
    with patch("ui.components.streaming_core.stream_chunks", return_value=iter(chunks)):
        list(_content_generator("q", "m", _SID, _MSG))
    final = SessionManager.get("stream_active_aux", None, _SID)
    assert final is not None
    assert final["citations"] == [_cit("x")]
