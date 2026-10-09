"""답변 생성 익스팬더(render_message) 엣지케이스 단위 검증.

렌더 경로 보장 (단순안 스펙):
- 완료 메시지는 본문 + 문서별 한 줄 요약 참조 목록만 렌더한다. 익스팬더
  ("Cited Sources" / "Sources & Reasoning" / "Thought Process"), 툴팁 주입,
  [1][2] 배지, 성능 지표 캡션("1.5s · 5→0 tok")은 노출되지 않는다.
- process.retrieved_count는 무시된다.
- thought 단독(문서/인용 없음) 턴은 본문만 렌더하고 익스팬더를 열지 않는다.
- 중단(cancelled) 메시지는 본문 + 참조 목록 + 중립 중단 캡션만 표시한다.
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


def _button_labels(mock_st: MagicMock) -> list[str]:
    labels = []
    for c in mock_st.button.call_args_list:
        if c.args:
            labels.append(str(c.args[0]))
    return labels


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
    """문서 턴은 익스팬더/지표 없이 참조 목록 행만 렌더한다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        documents=[_doc(3)],
        metrics={"total_time": 1.5, "tps": 10.0, "input_token_count": 5},
        process={"retrieved_count": 7},
        wrap_in_container=False,
    )
    assert mock_st.expander.call_args_list == []
    text = _all_text(mock_st)
    assert "1.5s" not in text
    assert "tok" not in text
    assert "Answer complete" not in text
    assert "references" not in text
    assert any("· p.3" in lbl for lbl in _button_labels(mock_st))


def test_metrics_uses_documents_length_over_retrieved_count():
    """하단 지표 캡션이 없고 retrieved_count가 노출되지 않습니다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        documents=[_doc(1), _doc(2)],
        metrics={"total_time": 1.5},
        process={"retrieved_count": 99},
        wrap_in_container=False,
    )
    text = _all_text(mock_st)
    assert "1.5s" not in text
    assert "99" not in text
    assert "references" not in text
    assert mock_st.expander.call_args_list == []


def test_detailed_thinking_skips_missing_keys_in_top_scores():
    """thought+문서 턴도 익스팬더 없이 참조 목록만 안전 렌더."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        thought="추론입니다.",
        documents=[_doc(3)],
        process={
            "steps": ["검색"],
            "top_scores": [
                {"section": "S1", "score": 0.912},
                {"bad": 1},  # 키 누락 → 무시 대상
            ],
            "perf": {"total_time": 1.0},
        },
        wrap_in_container=False,
    )
    assert mock_st.expander.call_args_list == []
    assert any("· p.3" in lbl for lbl in _button_labels(mock_st))


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
    """중단(+thought)이고 문서가 없어도 익스팬더 없이 중단 캡션만 렌더."""
    mock_st = _render(
        role="assistant",
        content="부분 답변입니다.",
        thought="미완료 추론.",
        process=None,
        wrap_in_container=False,
        cancelled=True,
    )
    assert mock_st.expander.call_args_list == []
    assert "Stopped" in _all_text(mock_st)


def test_cancelled_hides_thought_but_shows_references():
    """중단 시 참조 목록 + 중단 캡션만 표시하고 thought는 노출하지 않는다."""
    mock_st = _render(
        role="assistant",
        content="부분 답변입니다.",
        thought="미완료 추론.",
        documents=[_doc(1)],
        process={"steps": ["검색"]},
        wrap_in_container=False,
        cancelled=True,
    )
    assert mock_st.expander.call_args_list == []
    assert any("· p.1" in lbl for lbl in _button_labels(mock_st))
    assert "미완료 추론" not in _all_text(mock_st)
    assert "Stopped" in _all_text(mock_st)


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
    """완료된 어시스턴트 메시지는 익스팬더 대신 참조 목록을 렌더합니다.

    참고: 스트리밍 중 본문은 render_message가 아닌 _draw_streaming_message(전용
    슬롯) 경로로 그려지므로, render_message는 완료된 메시지만 처리한다.
    """
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        thought="추론입니다.",
        documents=[_doc(5)],
        msg_type="general",
        wrap_in_container=False,
    )
    assert mock_st.expander.call_args_list == []
    assert any("· p.5" in lbl for lbl in _button_labels(mock_st))


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
