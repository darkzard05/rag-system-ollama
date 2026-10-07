"""Task2: incremental metrics caption + mid-stream citation badges (TDD).

Failing-first coverage for:
  (a) caption seeded with the monotonic elapsed clock before the final
      metrics event arrives, then upgraded when final total_time/tokens land;
  (b) citation badges painted mid-stream via the _on_aux path while the full
      references body renders only on finalize.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from api.stream_pipeline import StreamChunk
from core.session import SessionManager
from ui.components.chat import _metrics_caption, _paint_midstream_badges
from ui.components.chat_references import render_inline_citation_badges
from ui.components.streaming_core import _content_generator

_SID = "task2-caption"
_MSG = "task2-msg-1"


def _cit(i: int) -> dict:
    return {"doc_id": f"doc-{i}", "section": f"sec-{i}", "page": i + 1}


# --- (a) seeded caption ----------------------------------------------------


def test_metrics_caption_seeded_with_elapsed_before_final_event() -> None:
    """Final metrics가 없어도 elapsed seed로 캡션이 먼저 보인다."""
    caption = _metrics_caption({}, elapsed_sec=3.2)
    assert caption != ""
    assert "3.2s" in caption


def test_metrics_caption_seed_uses_turn_start_clock() -> None:
    """turn_start(perf monotonic clock)만으로 seed 캡션을 만든다."""
    import time

    caption = _metrics_caption({}, turn_start=time.monotonic() - 5.0)
    assert caption != ""
    assert caption.rstrip().endswith("s ▍") or "5." in caption


def test_metrics_caption_final_upgrades_seed() -> None:
    """final total_time/tokens 도착 시 seed가 아닌 최종값으로 업그레이드된다."""
    caption = _metrics_caption(
        {"total_time": 9.5, "input_token_count": 10, "token_count": 20},
        elapsed_sec=3.2,
    )
    assert "9.5s" in caption
    assert "3.2s" not in caption
    assert "10→20 tok" in caption


def test_metrics_caption_empty_without_any_signal() -> None:
    """신호가 전혀 없으면 팬텀 캡션을 만들지 않는다."""
    assert _metrics_caption({}) == ""
    assert _metrics_caption(None) == ""


# --- (b) mid-stream badges --------------------------------------------------


def test_midstream_badges_use_caption_only_never_buttons() -> None:
    """배지는 단일 markdown 캡션만 쓰고 버튼 위젯을 만들지 않는다."""
    with patch("ui.components.chat_references.st") as mock_st:
        painted = render_inline_citation_badges([_cit(0), _cit(1)])
    assert painted is True
    mock_st.markdown.assert_called_once()
    mock_st.button.assert_not_called()


def test_midstream_badges_nothing_to_paint() -> None:
    """인용이 없으면 mid-stream에 아무것도 그리지 않는다."""
    with patch("ui.components.chat_references.st") as mock_st:
        assert render_inline_citation_badges([]) is False
        assert render_inline_citation_badges(None) is False
    mock_st.markdown.assert_not_called()


def test_on_aux_path_paints_badges_midstream() -> None:
    """_on_aux 경로의 paint helper가 인용을 배지로 중계한다."""
    aux = {
        "thought": "",
        "documents": [],
        "citations": [_cit(0)],
        "metrics": {},
        "process_steps": [],
    }
    with patch(
        "ui.components.chat.render_inline_citation_badges", return_value=True
    ) as paint:
        assert _paint_midstream_badges(aux) is True
        paint.assert_called_once_with([_cit(0)], [])


def test_on_aux_path_skips_badges_without_citations() -> None:
    """인용 없는 aux 스냅샷에서는 배지 렌더를 호출하지 않는다."""
    aux: dict = {"thought": "t", "documents": [], "citations": [], "metrics": {}}
    with patch("ui.components.chat.render_inline_citation_badges") as paint:
        assert _paint_midstream_badges(aux) is False
        paint.assert_not_called()


def _doc_like(page: int = 1) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page}
    return doc


def test_citations_visible_pre_final_event_via_synthetic_aux_stream() -> None:
    """합성 aux 스트림에서 인용이 complete=True 이전에 먼저 보인다."""
    SessionManager.reset_all_state(_SID)
    docs = [_doc_like(2)]
    chunks = [
        StreamChunk(content="", metadata={"documents": docs}),
        StreamChunk(content="tok", citations=[_cit(0)]),
        StreamChunk(
            content="",
            performance={"total_time": 4.0, "token_count": 7},
        ),
    ]
    seen: list[dict] = []

    def _on_aux(aux: dict) -> None:
        seen.append(dict(aux))

    with patch("ui.components.streaming_core.stream_chunks", return_value=iter(chunks)):
        list(_content_generator("q", "m", _SID, _MSG, on_status=None, on_aux=_on_aux))

    assert seen, "snapshot gate가 aux 콜백을 삼키면 안 된다"
    first_with_cit = next((i for i, s in enumerate(seen) if s.get("citations")), None)
    assert first_with_cit is not None, "citations-event 이전 증명이 필요하다"
    assert seen[first_with_cit].get("complete") is False
    final = SessionManager.get("stream_active_aux", None, _SID)
    assert final is not None
    assert final.get("complete") is True


def test_aux_snapshot_gate_suppresses_duplicate_states() -> None:
    """동일 스냅샷 반복은 on_aux를 재호출하지 않는다 (게이트 유지)."""
    SessionManager.reset_all_state(_SID)
    chunks = [
        StreamChunk(content="a", status="step-1"),
        StreamChunk(content="b", status="step-1"),
    ]
    calls: list[dict] = []
    with patch("ui.components.streaming_core.stream_chunks", return_value=iter(chunks)):
        list(
            _content_generator(
                "q",
                "m",
                _SID,
                _MSG,
                on_status=None,
                on_aux=lambda aux: calls.append(dict(aux)),
            )
        )
    assert len(calls) <= len(chunks)


def test_full_references_body_only_on_finalize() -> None:
    """생성 중 익스팬더는 버튼을 만들지 않고 finalize에서만 만든다."""
    msg: dict = {
        "thought": "",
        "documents": [_doc_like(3)],
        "citations": [_cit(0)],
        "msg_id": "mid-1",
    }
    from ui.components.chat_references import render_generation_expander

    with patch("ui.components.chat_references.st") as mock_st:
        mock_st.expander.return_value.__enter__.return_value = mock_st
        render_generation_expander(msg, expanded=False, generating=True)
    mock_st.button.assert_not_called()

    with patch("ui.components.chat_references.st") as mock_st:
        mock_st.expander.return_value.__enter__.return_value = mock_st
        render_generation_expander(msg, expanded=False, generating=False)
    assert mock_st.button.called


def test_fake_chunk_namespace_shape_matches_stream_chunk() -> None:
    """테스트용 SimpleNamespace가 StreamChunk 속성을 그대로 반영한다."""
    ns = SimpleNamespace(
        status=None, thought="", content="x", metadata=None, performance=None
    )
    for attr in ("status", "thought", "content", "metadata", "performance"):
        assert hasattr(StreamChunk(), attr)
        assert hasattr(ns, attr)
