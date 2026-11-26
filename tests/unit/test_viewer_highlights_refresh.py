from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components.chat import _current_page_annotations, _should_skip_final_rerun
from ui.components.reference_plan import _make_prewarm_done_callback
from ui.widget_keys import pdf_viewer_key


def test_pdf_viewer_key_signature_branching():
    """정상 케이스 1: pdf_viewer_key 하위 호환 및 주석 시그니처 분기 검증"""
    assert pdf_viewer_key("hash1", 1) == "pdf_v8_hash1_1"
    assert pdf_viewer_key("hash1", 1, "abcd1234") == "pdf_v8_hash1_1_abcd1234"


def test_other_page_citations_skip_rerun():
    """정상 케이스 2: 타 페이지 인용 시 불필요한 리런 스킵 유지"""
    sid = SessionManager.get_session_id()
    SessionManager.set("file_hash", "hash1", session_id=sid)
    SessionManager.set("current_page", 1, session_id=sid)
    SessionManager.set(
        "pdf_annotations",
        {
            "file_hash": "hash1",
            "annotations": [
                {
                    "page": 3,
                    "x": 10,
                    "y": 10,
                    "width": 20,
                    "height": 10,
                    "color": "red",
                }
            ],
        },
        session_id=sid,
    )

    scoped = _current_page_annotations(sid)
    assert scoped == []

    should_skip = _should_skip_final_rerun(
        cancelled=False,
        error_text=None,
        rendered_while_generating=False,
        current_page_annotations_changed=False,
    )
    assert should_skip is True


def test_current_page_annotations_changed_triggers_rerun():
    """버그 재현 케이스 1: 현재 페이지 하이라이트 동기 반영 및 리런 허용 검증"""
    sid = SessionManager.get_session_id()
    SessionManager.set("file_hash", "hash1", session_id=sid)
    SessionManager.set("current_page", 1, session_id=sid)

    # 초기 주석 없음
    SessionManager.set(
        "pdf_annotations", {"file_hash": "hash1", "annotations": []}, session_id=sid
    )
    prev = _current_page_annotations(sid)
    assert prev == []

    # finalize 이후 현재 페이지(page=1) 주석 추가 시뮬레이션
    SessionManager.set(
        "pdf_annotations",
        {
            "file_hash": "hash1",
            "annotations": [
                {
                    "page": 1,
                    "x": 10,
                    "y": 10,
                    "width": 20,
                    "height": 10,
                    "color": "red",
                }
            ],
        },
        session_id=sid,
    )
    next_ann = _current_page_annotations(sid)
    assert len(next_ann) == 1

    changed = next_ann != prev
    assert changed is True

    should_skip = _should_skip_final_rerun(
        cancelled=False,
        error_text=None,
        rendered_while_generating=False,
        current_page_annotations_changed=changed,
    )
    assert should_skip is False


def test_async_prewarm_callback_triggers_rerun_when_matching():
    """버그 재현 케이스 2: 비동기 prewarm 완료 시 현재 페이지 리런 요청 검증"""
    sid = SessionManager.get_session_id()
    SessionManager.set("file_hash", "hash1", session_id=sid)
    SessionManager.set("current_page", 2, session_id=sid)
    SessionManager.set("is_generating_answer", False, session_id=sid)

    future = MagicMock()
    future.result.return_value = [
        {"page": 2, "x": 5, "y": 5, "width": 10, "height": 10}
    ]

    callback = _make_prewarm_done_callback(sid, "doc_sig_1")

    with patch("common.async_worker.request_session_rerun") as mock_request_rerun:
        callback(future)
        mock_request_rerun.assert_called_once_with(sid)

    # annotations가 현재 페이지(2)와 일치하지 않는 경우
    future_other = MagicMock()
    future_other.result.return_value = [
        {"page": 5, "x": 5, "y": 5, "width": 10, "height": 10}
    ]
    callback_other = _make_prewarm_done_callback(sid, "doc_sig_2")

    with patch("common.async_worker.request_session_rerun") as mock_request_rerun:
        callback_other(future_other)
        mock_request_rerun.assert_not_called()

    # is_generating_answer가 True인 경우
    SessionManager.set("is_generating_answer", True, session_id=sid)
    future_gen = MagicMock()
    future_gen.result.return_value = [
        {"page": 2, "x": 5, "y": 5, "width": 10, "height": 10}
    ]
    callback_gen = _make_prewarm_done_callback(sid, "doc_sig_3")

    with patch("common.async_worker.request_session_rerun") as mock_request_rerun:
        callback_gen(future_gen)
        mock_request_rerun.assert_not_called()


def test_prewarm_callback_stamps_hash_captured_at_submit_time():
    """새 문서 업로드 후 늦게 끝난 콜백이 이전 문서 좌표를 새 해시로 붙이지 않는다."""
    sid = SessionManager.get_session_id()
    SessionManager.set("file_hash", "hash-new", session_id=sid)
    SessionManager.set("is_generating_answer", False, session_id=sid)

    future = MagicMock()
    future.result.return_value = [
        {"page": 1, "x": 5, "y": 5, "width": 10, "height": 10}
    ]
    callback = _make_prewarm_done_callback(sid, "sig_captured_hash", "hash-old")
    callback(future)

    stored = SessionManager.get("pdf_annotations", None, sid)
    assert stored["file_hash"] == "hash-old"


def test_empty_prewarm_result_clears_stale_highlights():
    """빈 주석 결과도 화면을 갱신해 이전 턴 하이라이트를 지워야 한다."""
    sid = SessionManager.get_session_id()
    SessionManager.set("file_hash", "hash-old", session_id=sid)
    SessionManager.set("current_page", 2, session_id=sid)
    SessionManager.set("is_generating_answer", False, session_id=sid)
    SessionManager.set(
        "pdf_annotations",
        {
            "file_hash": "hash-old",
            "annotations": [{"page": 2, "x": 5, "y": 5, "width": 10, "height": 10}],
        },
        session_id=sid,
    )

    future = MagicMock()
    future.result.return_value = []
    callback = _make_prewarm_done_callback(sid, "sig_empty_result", "hash-old")

    with patch("common.async_worker.request_session_rerun") as mock_request_rerun:
        callback(future)
        mock_request_rerun.assert_called_once_with(sid)

    stored = SessionManager.get("pdf_annotations", None, sid)
    assert stored["annotations"] == []
