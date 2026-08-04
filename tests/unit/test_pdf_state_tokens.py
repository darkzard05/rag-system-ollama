"""PDF 뷰어 참조페이지 이동 토큰(_resolve_pdf_state) 의미론 검증.

pdf_target_page 토큰의 일회성 소비, stale 판정, 클램핑, 자가 정리(self-clean)
동작을 SessionManager 테스트 세션 + monkeypatch로 검증한다. 실제 PDF가 필요하지
않도록 _get_pdf_total_pages를 모듈 레벨에서 직접 패치한다 (cache 데코레이터 회피).
"""

import time

import pytest

import ui.components.viewer as viewer_module
from core.session import SessionManager

SID = "test_pdf_tokens"


@pytest.fixture
def pdf_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """_resolve_pdf_state 실행 가능 상태 + fake st.session_state를 제공한다."""
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("pdf_file_path", "dummy.pdf", SID)
    SessionManager.set("file_hash", "testhash", SID)
    # cache 데코레이터(@safe_cache_data)를 우회: 모듈 이름을 직접 교체.
    monkeypatch.setattr(viewer_module, "_get_pdf_total_pages", lambda path: 10)
    fake_session_state: dict[str, object] = {}
    monkeypatch.setattr(viewer_module.st, "session_state", fake_session_state)
    return fake_session_state


def _set_token(token: object) -> None:
    SessionManager.set("pdf_target_page", token, SID)


def test_manual_token_consumed_once(pdf_state: dict[str, object]) -> None:
    """수동 토큰: 첫 호출에서 소비되며(삭제) 이후 재점프하지 않는다."""
    _set_token({"page": 5, "source": "manual", "ts": time.time()})

    first = viewer_module._resolve_pdf_state()
    assert first is not None
    assert first["current_page"] == 5
    # 일회성 소비: 호출 후 토큰 삭제 확인
    assert SessionManager.get("pdf_target_page", session_id=SID) is None

    # 두 번째 호출은 pdf_nav_input_v6로 폴스루, 재점프 없이 동일 페이지 유지
    second = viewer_module._resolve_pdf_state()
    assert second is not None
    assert second["current_page"] == 5
    assert pdf_state["pdf_nav_input_v6"] == 5


def test_auto_token_consumed_when_no_manual_nav(
    pdf_state: dict[str, object],
) -> None:
    """수동 네비게이션이 없으면 auto 토큰도 그대로 소비된다."""
    _set_token({"page": 7, "source": "auto", "ts": time.time()})

    state = viewer_module._resolve_pdf_state()
    assert state is not None
    assert state["current_page"] == 7
    assert SessionManager.get("pdf_target_page", session_id=SID) is None


def test_stale_auto_token_discarded_after_manual_nav(
    pdf_state: dict[str, object],
) -> None:
    """사용자 수동 네비게이션이 토큰보다 최신이면 토큰은 폐기되고 폴스루한다."""
    _set_token({"page": 7, "source": "auto", "ts": 100.0})
    SessionManager.set("manual_nav_ts", 200.0, SID)
    pdf_state["pdf_nav_input_v6"] = 3

    state = viewer_module._resolve_pdf_state()
    assert state is not None
    # 점프하지 않고 수동 입력 페이지로 폴스루
    assert state["current_page"] == 3
    assert state["current_page"] != 7
    assert SessionManager.get("pdf_target_page", session_id=SID) is None


def test_legacy_int_token_supported(pdf_state: dict[str, object]) -> None:
    """레거시 int 토큰은 source=manual 취급되어 소비된다."""
    _set_token(9)

    state = viewer_module._resolve_pdf_state()
    assert state is not None
    assert state["current_page"] == 9
    assert SessionManager.get("pdf_target_page", session_id=SID) is None


def test_token_page_clamped_to_total_pages(
    pdf_state: dict[str, object],
) -> None:
    """초과/0페이지 토큰은 total_pages(10)와 1 사이로 클램프된다."""
    _set_token({"page": 99, "source": "manual", "ts": 0})
    high_state = viewer_module._resolve_pdf_state()
    assert high_state is not None
    assert high_state["current_page"] == 10
    assert SessionManager.get("pdf_target_page", session_id=SID) is None

    _set_token({"page": 0, "source": "manual", "ts": 0})
    low_state = viewer_module._resolve_pdf_state()
    assert low_state is not None
    assert low_state["current_page"] == 1


def test_malformed_dict_token_self_cleans(
    pdf_state: dict[str, object],
) -> None:
    """page 누락 dict 토큰은 크래시 없이 폐기되고 폴스루한다."""
    _set_token({"source": "manual"})
    pdf_state["pdf_nav_input_v6"] = 2

    state = viewer_module._resolve_pdf_state()
    assert state is not None
    assert state["current_page"] == 2
    assert SessionManager.get("pdf_target_page", session_id=SID) is None
