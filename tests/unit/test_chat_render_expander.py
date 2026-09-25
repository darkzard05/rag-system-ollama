"""답변 생성 익스팬더(render_message) 엣지케이스 단위 검증.

렌더 경로 보장 (신규 스펙):
- 완료 후 하단 상태줄은 핵심 메트릭만 표시 ("1.5s · 5→0 tok" 형식,
  모델명 제외). "Answer complete" 캡션, 참조 개수("N references"),
  페이지 미리보기("p.X")는 노출되지 않으며 process.retrieved_count는 무시된다
- 완료 후 영속 익스팬더 라벨은 내용 조합에 따라 결정
  (문서만 → "Cited Sources", 추론+문서 → "Sources & Reasoning",
  추론만 → "Thought Process")
- 참조 전용 익스팬더는 top_scores/thought 키 누락 시 KeyError 없이 안전 렌더
  (process는 익스팬더 조건에 사용되지 않음)
- process=None / 빈 process / thought·문서·인용 모두 없음 시 익스팬더 미노출
  (빈 본문 방지). 익스팬더는 thought/documents/citations 중 하나라도 있을 때만
  열리며 expanded = is_latest and bool(thought or documents)
- 중단(cancelled) 메시지는 thought를 그대로 보여주고 하단에
  "Stopped · Partial answer preserved" 캡션을 표시한다
"""

from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components.chat import _draw_streaming_message, render_message


def _render(**kwargs) -> MagicMock:
    """render_message를 호출하며 ui.components.chat.st를 mock 처리합니다.

    expander/popover/container/chat_message/columns 컨텍스트 매니저는 모두 동일
    mock_st를 반환하므로 본문/캡션/익스팬더 호출을 한 mock에서 관찰할 수 있습니다.
    """
    with (
        patch("ui.components.chat.st") as mock_st,
        patch("ui.components.chat_references.st", mock_st),
    ):
        mock_st.container.return_value.__enter__.return_value = mock_st
        mock_st.chat_message.return_value.__enter__.return_value = mock_st
        mock_st.popover.return_value.__enter__.return_value = mock_st
        mock_st.expander.return_value.__enter__.return_value = mock_st
        # st.columns(min(len(pages), 5)) 반환값을 길이 1의 열 리스트로 만들어
        # 참조 popover의 idx % len(cols) ZeroDivision을 방지.
        mock_st.columns.return_value = [mock_st]
        render_message(**kwargs)
    return mock_st


def _expander_labels(mock_st: MagicMock) -> list[str]:
    return [c.args[0] for c in mock_st.expander.call_args_list]


def _all_text(mock_st: MagicMock) -> str:
    parts = []
    for call in mock_st.markdown.call_args_list + mock_st.caption.call_args_list:
        if call.args:
            parts.append(str(call.args[0]))
    return "\n".join(parts)


def _doc(page: int = 1) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page}
    doc.page_content = "참조 본문"
    return doc


def test_metrics_shows_document_count():
    """완료 상태줄은 핵심 메트릭만 표시하고 문서 전용 익스팬더를 엽니다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        documents=[_doc(3)],
        metrics={"total_time": 1.5, "tps": 10.0, "input_token_count": 5},
        process={"retrieved_count": 7},
        wrap_in_container=False,
    )
    assert "Cited Sources" in _expander_labels(mock_st)
    text = _all_text(mock_st)
    assert "1.5s" in text
    assert "5→0 tok" in text
    assert "Answer complete" not in text
    assert "references" not in text
    assert "p.3" not in text


def test_metrics_uses_documents_length_over_retrieved_count():
    """하단 캡션은 메트릭 전용이며 process.retrieved_count가 노출되지 않습니다.

    retrieved_count는 상태줄에 노출되지 않으므로 무시되고 핵심 메트릭만
    표시됩니다.
    """
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        documents=[_doc(1), _doc(2)],
        metrics={"total_time": 1.5},
        process={"retrieved_count": 99},
        wrap_in_container=False,
    )
    text = _all_text(mock_st)
    assert "1.5s" in text
    assert "99" not in text
    assert "references" not in text


def test_detailed_thinking_skips_missing_keys_in_top_scores():
    """참조 전용 익스팬더는 top_scores/thought 등이 없어도 안전 렌더."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        thought="추론입니다.",
        documents=[_doc(3)],
        process={
            "steps": ["검색"],
            "top_scores": [
                {"section": "S1", "score": 0.912},
                {"bad": 1},  # 키 누락 → 스킵 대상
            ],
            "perf": {"total_time": 1.0},
        },
        wrap_in_container=False,
    )
    assert "Sources & Reasoning" in _expander_labels(mock_st)


def test_no_detailed_thinking_when_process_none():
    """process=None이면 익스팬더를 열지 않고 크래시하지 않습니다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        process=None,
        wrap_in_container=False,
    )
    assert mock_st.expander.call_args_list == []


def test_no_detailed_thinking_when_cancelled_without_process():
    """중단(+thought)이고 process가 없어도 추론 익스팬더는 렌더됩니다."""
    mock_st = _render(
        role="assistant",
        content="부분 답변입니다.",
        thought="미완료 추론.",
        process=None,
        wrap_in_container=False,
        cancelled=True,
    )
    assert "Thought Process" in _expander_labels(mock_st)
    assert "Stopped · Partial answer preserved" in _all_text(mock_st)


def test_cancelled_hides_thought_but_shows_references():
    """중단 시에도 thought와 참조가 함께 익스팬더에 렌더되고 중단 캡션이 표시됩니다."""
    mock_st = _render(
        role="assistant",
        content="부분 답변입니다.",
        thought="미완료 추론.",
        documents=[_doc(1)],
        process={"steps": ["검색"]},
        wrap_in_container=False,
        cancelled=True,
    )
    assert "Sources & Reasoning" in _expander_labels(mock_st)
    assert "미완료 추론" in _all_text(mock_st)
    assert "Stopped · Partial answer preserved" in _all_text(mock_st)


def test_empty_expander_suppressed_without_content():
    """process가 빈 구조이고 thought 없으면 익스팬더를 열지 않습니다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        process={"steps": [], "sections": [], "top_scores": [], "perf": {}},
        wrap_in_container=False,
    )
    assert mock_st.expander.call_args_list == []


def test_completed_assistant_message_opens_expander():
    """완료된 어시스턴트 메시지에 참조가 있으면 영속 익스팬더를 렌더합니다.

    참고: 스트리밍 중 본문은 render_message가 아닌 _draw_streaming_message(전용
    슬롯) 경로로 그려지므로, render_message는 완료된 메시지만 처리한다. 확정
    메시지의 추론·참조 부가 정보는 내용 조합에 따른 익스팬더에 수납된다.
    """
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        thought="추론입니다.",
        documents=[_doc(5)],
        msg_type="general",
        wrap_in_container=False,
    )
    assert "Sources & Reasoning" in _expander_labels(mock_st)


# ---------------------------------------------------------------------------
# T14: UX-3 중립 잔존 처리 — _draw_streaming_message 엣지케이스
# ---------------------------------------------------------------------------


def test_draw_streaming_message_empty_placeholder_shows_neutral_caption() -> None:
    """content 없는 스트리밍 플레이스홀더 → t(status_stopped) 캡션 + retry CTA, expander/error 없음."""
    from ui.strings import t

    sid = "render_empty_stream"
    SessionManager.reset_all_state(sid)

    with (
        patch("ui.components.chat.st") as mock_st,
        patch("ui.components.chat.render_generation_expander") as mock_expander,
    ):
        _draw_streaming_message(
            {"role": "assistant", "content": "", "msg_type": "streaming"},
            sid,
        )

    expected = t("status_stopped")
    texts = [call.args[0] for call in mock_st.caption.call_args_list if call.args]
    assert any(expected in text for text in texts)
    assert mock_st.button.called
    mock_expander.assert_not_called()
    mock_st.error.assert_not_called()


def test_draw_streaming_message_with_content_keeps_generating_expander() -> None:
    """content 있는 스트리밍 메시지 → expander generating=True 유지 (회귀 가드)."""
    sid = "render_content_stream"
    SessionManager.reset_all_state(sid)

    with (
        patch("ui.components.chat.st") as mock_st,
        patch("ui.components.chat.render_generation_expander") as mock_expander,
    ):
        _draw_streaming_message(
            {"role": "assistant", "content": "부분 답변", "msg_type": "streaming"},
            sid,
        )

    from ui.strings import t

    mock_expander.assert_called_once()
    assert mock_expander.call_args.kwargs["generating"] is True
    neutral = [
        call.args[0]
        for call in mock_st.caption.call_args_list
        if call.args and t("status_stopped") in call.args[0]
    ]
    assert neutral == []
