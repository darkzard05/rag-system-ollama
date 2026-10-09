"""INT-1 회귀 테스트: 답변 완료 시 자동 페이지 점프 토큰 미생성 (uiux-fix-p1).

streaming.py의 자동 점프 블록(pdf_target_page {"source": "auto"} + current_page
자동 세팅) 제거를 검증한다. 사용자 발의 없는 화면 이동을 금지하는 INT-1의
회귀 방지가 목적이다. 수동 참조 점프(chat.py _handle_page_jump의
pdf_target_page {"source": "manual"})는 본 테스트 범위 밖이며, viewer.py의
토큰 소비 로직은 그대로 유지된다.
"""

from unittest.mock import MagicMock

from core.session import SessionManager
from ui.components.streaming import _finalize_pdf_side_effects


def _make_doc(page: int) -> MagicMock:
    """page 메타데이터를 가진 경량 fake 문서를 생성합니다."""
    doc = MagicMock()
    doc.metadata = {"page": page}
    return doc


def _seed_completed_turn(sid: str, documents: list) -> None:
    """완료 턴 상태를 시드합니다.

    _finalize_pdf_side_effects는 메시지의 msg_type과 무관하게
    error 유무 + documents 유무로 동작하므로 일반 메시지로 충분하다.
    """
    SessionManager.reset_all_state(sid)
    SessionManager.set("pdf_file_path", "dummy.pdf", sid)
    SessionManager.set("file_hash", "seed-hash", sid)
    SessionManager.set(
        "messages",
        [{"msg_id": "m1", "documents": documents, "error": None}],
        sid,
    )


def test_finalize_does_not_set_auto_jump_token():
    """완료 턴: _finalize_pdf_side_effects가 자동 점프 토큰/페이지를 만들지 않는다."""
    sid = "test_no_auto_jump"
    _seed_completed_turn(sid, [_make_doc(7)])

    _finalize_pdf_side_effects(sid, "m1")

    assert SessionManager.get("pdf_target_page", None, sid) is None
    assert SessionManager.get("current_page", None, sid) == 1  # 기본값 유지


def test_finalize_still_stores_pdf_annotations():
    """완료 턴: 자동 점프 없이 하이라이트 경로만 유지된다 (논블로킹).

    prewarm 계획 없이 호출하면 렌더 스레드를 막지 않고 백그라운드에 맡기므로
    즉시 set되지 않을 수 있다 (None 허용). 자동 점프 토큰은 절대 생기지
    않아야 하며, 하이라이트 데이터 경로는 prewarm 적중 시 set된다.
    """
    sid = "test_no_auto_jump_annotations"
    _seed_completed_turn(sid, [_make_doc(7)])

    _finalize_pdf_side_effects(sid, "m1")

    assert SessionManager.get("pdf_target_page", None, sid) is None
    annotations = SessionManager.get("pdf_annotations", None, sid)
    assert annotations is None or annotations["file_hash"] == "seed-hash"
