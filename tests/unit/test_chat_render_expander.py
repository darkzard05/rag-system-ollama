"""답변 생성 익스팬더(render_message) 엣지케이스 단위 검증.

렌더 경로 보장:
- Metrics "Retrieved"는 데드 키(retrieved_chunks) 대신 process.retrieved_count 사용
- 완료 후 영속 익스팬더는 라벨 "Answer details"를 사용 (스트리밍 박스의
  스피너 status_text="Answer generation"과는 별개). Metrics는 별도 익스팬더가
  아닌 동일 익스팬더 내 "Retrieved: N chunks" 캡션으로 통합됨.
- Answer details 익스팬더는 top_scores 항목 키 누락 시 KeyError 없이 안전 렌더
- process=None / 빈 process / cancelled+thought(내용 없음) 시 익스팬더 미노출(빈 본문 방지)
"""

from unittest.mock import MagicMock, patch

from ui.components.chat import render_message


def _render(**kwargs) -> MagicMock:
    """render_message를 호출하며 ui.components.chat.st를 mock 처리합니다.

    expander/popover/container/chat_message/columns 컨텍스트 매니저는 모두 동일
    mock_st를 반환하므로 본문/캡션/익스팬더 호출을 한 mock에서 관찰할 수 있습니다.
    """
    with patch("ui.components.chat.st") as mock_st:
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
    """Metrics 익스팬더는 documents가 있을 때만 열리며 실제 청크 수를 표시합니다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        documents=[_doc(3)],
        metrics={"total_time": 1.5, "tps": 10.0, "input_token_count": 5},
        process={"retrieved_count": 7},
        wrap_in_container=False,
    )
    assert "Answer details" in _expander_labels(mock_st)
    assert "Retrieved: 1 chunks" in _all_text(mock_st)


def test_metrics_uses_documents_length_over_retrieved_count():
    """documents가 있으면 process.retrieved_count 대신 len(documents)를 우선합니다.

    데드 키 metrics["retrieved_count"] 의존성이 제거되었으므로 retrieved_count는
    무시되고 실제 문서 수가 표시됩니다.
    """
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        documents=[_doc(1), _doc(2)],
        metrics={"total_time": 1.5},
        process={"retrieved_count": 99},
        wrap_in_container=False,
    )
    assert "Retrieved: 2 chunks" in _all_text(mock_st)
    assert "Retrieved: 99 chunks" not in _all_text(mock_st)


def test_detailed_thinking_skips_missing_keys_in_top_scores():
    """top_scores 항목에 section/score가 없어도 KeyError 없이 안전 렌더."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        thought="추론입니다.",
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
    assert "Answer details" in _expander_labels(mock_st)
    # 유효 항목은 포맷 유지, 잘못된 항목은 렌더되지 않아야 함
    assert "S1 0.912" in _all_text(mock_st)
    assert "bad" not in _all_text(mock_st)


def test_no_detailed_thinking_when_process_none():
    """process=None이면 익스팬더를 열지 않고 크래시하지 않습니다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        process=None,
        wrap_in_container=False,
    )
    assert "Answer details" not in _expander_labels(mock_st)


def test_no_detailed_thinking_when_cancelled_without_process():
    """중단(+thought)이고 process가 없으면 익스팬더를 열지 않습니다."""
    mock_st = _render(
        role="assistant",
        content="부분 답변입니다.",
        thought="미완료 추론.",
        process=None,
        wrap_in_container=False,
        cancelled=True,
    )
    assert "Answer details" not in _expander_labels(mock_st)


def test_cancelled_hides_thought_but_shows_process():
    """중단 시 steps는 익스팬더에 노출되나 thought 텍스트는 감춰집니다."""
    mock_st = _render(
        role="assistant",
        content="부분 답변입니다.",
        thought="미완료 추론.",
        process={"steps": ["검색"]},
        wrap_in_container=False,
        cancelled=True,
    )
    assert "Answer details" in _expander_labels(mock_st)
    assert "미완료 추론" not in _all_text(mock_st)


def test_empty_expander_suppressed_without_content():
    """process가 빈 구조이고 thought 없으면 익스팬더를 열지 않습니다."""
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        process={"steps": [], "sections": [], "top_scores": [], "perf": {}},
        wrap_in_container=False,
    )
    assert "Answer details" not in _expander_labels(mock_st)


def test_completed_assistant_message_opens_expander():
    """완료된 어시스턴트 메시지(thought 포함, 최종 msg_type)는 영속 익스팬더를 엽니다.

    참고: 스트리밍 중 본문은 render_message가 아닌 _draw_streaming_message(전용
    슬롯) 경로로 그려지므로, render_message는 완료된 메시지만 처리한다. 따라서
    확정 메시지에 thought/문서/메트릭이 있으면 항상 익스팬더가 열린다.
    """
    mock_st = _render(
        role="assistant",
        content="답변입니다.",
        thought="추론입니다.",
        msg_type="general",
        wrap_in_container=False,
    )
    assert "Answer details" in _expander_labels(mock_st)
