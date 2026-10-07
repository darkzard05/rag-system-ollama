"""Phase 1: prewarm submit 정책 단위 테스트.

PREWARM_SUBMIT_ON = {first citations_add, 5s-elapsed-once, final}.
중간 증분은 submit 하지 않는다. ``_clock`` 을 패치해 5초 경과를 시뮬레이션한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

from api.stream_pipeline import StreamChunk
from core.session import SessionManager
from ui.components import streaming_core
from ui.components.streaming_core import _content_generator

_SID = "prewarm-sid"
_MSG = "prewarm-msg"


def _cit(i: int) -> dict:
    return {
        "doc_id": f"doc-{i}",
        "page": i + 1,
        "section": f"s{i}",
        "text_span": f"t{i}",
    }


def _timed_chunks(
    chunks: list[StreamChunk], times: list[float], t: list[float]
) -> Iterator[StreamChunk]:
    """각 청크를 소비하기 전에 fake clock 시각을 설정한다."""
    for chunk, ts in zip(chunks, times, strict=True):
        t[0] = ts
        yield chunk


def test_prewarm_submits_first_elapsed_final_only() -> None:
    SessionManager.reset_all_state(_SID)
    t = [1000.0]
    chunks = [
        StreamChunk(content="tok"),  # t=1000
        StreamChunk(
            citations_add={
                "items": [_cit(0)],
                "base_index": 0,
                "arr_len": 1,
                "attempt": 0,
            }
        ),  # t=1001 → first_add
        StreamChunk(
            citations_add={
                "items": [_cit(1)],
                "base_index": 1,
                "arr_len": 2,
                "attempt": 0,
            }
        ),  # t=1002 → skip
        StreamChunk(
            citations_add={
                "items": [_cit(2)],
                "base_index": 2,
                "arr_len": 3,
                "attempt": 0,
            }
        ),  # t=1006 → elapsed_5s
        StreamChunk(
            citations_add={
                "items": [_cit(3)],
                "base_index": 3,
                "arr_len": 4,
                "attempt": 0,
            }
        ),  # t=1007 → skip
        StreamChunk(citations=[_cit(0), _cit(1), _cit(2), _cit(3)]),  # t=1008 → final
    ]
    times = [1000.0, 1001.0, 1002.0, 1006.0, 1007.0, 1008.0]
    prewarm_calls: list[tuple] = []

    def _fake_prewarm(sid, msg_id, documents, citations):
        prewarm_calls.append((sid, msg_id, list(documents), list(citations)))

    with (
        patch.object(
            streaming_core,
            "stream_chunks",
            return_value=_timed_chunks(chunks, times, t),
        ),
        patch.object(
            streaming_core, "_note_reference_prewarm", side_effect=_fake_prewarm
        ),
        patch.object(streaming_core, "_clock", side_effect=lambda: t[0]),
    ):
        list(_content_generator("q", "m", _SID, _MSG))

    assert len(prewarm_calls) == 3
    final = SessionManager.get("stream_active_aux", None, _SID)
    assert final is not None
    assert len(final["citations"]) == 4


def test_prewarm_first_add_after_5s_consumes_elapsed_flag() -> None:
    """첫 citations_add가 이미 5초 이후면 elapsed 플래그를 소비해 중복 submit 방지."""
    SessionManager.reset_all_state(_SID)
    t = [1000.0]
    chunks = [
        StreamChunk(content="tok"),
        StreamChunk(
            citations_add={
                "items": [_cit(0)],
                "base_index": 0,
                "arr_len": 1,
                "attempt": 0,
            }
        ),
        StreamChunk(citations=[_cit(0)]),
    ]
    times = [1000.0, 1006.0, 1008.0]
    prewarm_calls: list[tuple] = []

    def _fake_prewarm(sid, msg_id, documents, citations):
        prewarm_calls.append((sid, msg_id))

    with (
        patch.object(
            streaming_core,
            "stream_chunks",
            return_value=_timed_chunks(chunks, times, t),
        ),
        patch.object(
            streaming_core, "_note_reference_prewarm", side_effect=_fake_prewarm
        ),
        patch.object(streaming_core, "_clock", side_effect=lambda: t[0]),
    ):
        list(_content_generator("q", "m", _SID, _MSG))

    # first_add(1) + final(1) = 2 (elapsed는 first_add에서 소비됨)
    assert len(prewarm_calls) == 2
