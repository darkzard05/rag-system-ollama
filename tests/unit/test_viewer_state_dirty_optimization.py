import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import ui.components.viewer as viewer_module
from core.session import SessionManager

SID = "test_viewer_dirty_opt"


@pytest.fixture(autouse=True)
def setup_session(monkeypatch: pytest.MonkeyPatch):
    SessionManager.reset()
    SessionManager.init_session(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("pdf_file_path", "dummy.pdf", SID)
    SessionManager.set("file_hash", "testhash", SID)
    # 업로드 시점에 사전 계산되는 total_pages(프로덕션 불변식)를 시드한다.
    # 폴백 캐싱이 dirty 플래그를 남기지 않도록 디스크 조회 자체를 우회한다.
    SessionManager.set("total_pages", 10, SID)
    monkeypatch.setattr(viewer_module, "_get_pdf_total_pages", lambda path: 10)
    fake_session_state: dict[str, object] = {}
    monkeypatch.setattr(viewer_module.st, "session_state", fake_session_state)
    yield
    SessionManager.reset()


def test_resolve_pdf_state_does_not_dirty_when_page_unchanged():
    """페이지 번호가 정상 범위 내에서 동일하게 유지될 경우 SessionManager에 dirty 플래그가 남지 않아야 함."""
    # Given: current_page가 3으로 설정되고 dirty 플래그가 클리어된 상태
    SessionManager.set("current_page", 3, SID)
    # _dirty_keys 초기화 (동기화 완료 상태 시뮬레이션)
    with SessionManager._acquire_lock(SID):
        SessionManager._fallback_sessions[SID]["_dirty_keys"].clear()

    assert SessionManager.has_pending_ui_sync(SID) is False

    # When: _resolve_pdf_state 실행
    state = viewer_module._resolve_pdf_state()

    # Then: 뷰어 상태는 정상 반환되나 불필요한 set 호출이 없어 dirty 플래그가 여전히 False여야 함
    assert state is not None
    assert state["current_page"] == 3
    assert SessionManager.has_pending_ui_sync(SID) is False, (
        "_resolve_pdf_state가 페이지 변경 없이 SessionManager.set을 호출하여 dirty 상태를 유발함"
    )


def test_resolve_pdf_state_dirties_when_page_clamped():
    """범위를 벗어난 페이지 번호가 클램핑되어 실제 값이 변경될 때는 정상적으로 set 호출 및 dirty 반영."""
    # Given: total_pages(10)를 초과하는 15페이지로 설정 후 dirty 클리어
    SessionManager.set("current_page", 15, SID)
    with SessionManager._acquire_lock(SID):
        SessionManager._fallback_sessions[SID]["_dirty_keys"].clear()

    assert SessionManager.has_pending_ui_sync(SID) is False

    # When: _resolve_pdf_state 실행 (10으로 클램핑되어야 함)
    state = viewer_module._resolve_pdf_state()

    # Then: 값이 15 -> 10으로 변경되었으므로 set이 호출되어 dirty 플래그가 True가 되어야 함
    assert state is not None
    assert state["current_page"] == 10
    assert SessionManager.has_pending_ui_sync(SID) is True
