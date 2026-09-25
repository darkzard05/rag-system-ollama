"""Viewer nav clamp + narrow fragment deps 가드 (TASK 1, TDD failing-first).

수호 조건:
- total None이면 _on_page_change는 no-op (current_page 불변).
- number 입력은 1<=page<=total 로 클램프된다.
- 렌더당 resolve는 _resolve_pdf_state 1회뿐이며, 콜백이 total을 재조회하지 않고
  렌더에서 캐시된 total을 args로 전달받는다 (fragment deps = pdf_id+page).
- st.rerun 추가 금지 (회귀 가드).
"""

import inspect

import pytest

import ui.components.common as common_module
import ui.components.viewer as viewer_module
from core.session import SessionManager
from ui.widget_keys import PDF_NAV_INPUT_KEY

SID = "test_viewer_nav_clamp"


@pytest.fixture
def nav_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """SessionManager 세션 + viewer/common 공용 fake session_state를 제공한다."""
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("current_page", 5, SID)
    fake_session_state: dict[str, object] = {PDF_NAV_INPUT_KEY: 5}
    monkeypatch.setattr(viewer_module.st, "session_state", fake_session_state)
    monkeypatch.setattr(common_module.st, "session_state", fake_session_state)
    return fake_session_state


def _current_page() -> int:
    return int(SessionManager.get("current_page", 1))


# --- (a) total None이면 no-op -------------------------------------------------


def test_on_page_change_noop_when_total_none(
    nav_state: dict[str, object],
) -> None:
    """total을 알 수 없으면 입력 변경을 무시한다 (no-op)."""
    nav_state[PDF_NAV_INPUT_KEY] = 99
    viewer_module._on_page_change(None)
    assert _current_page() == 5
    assert nav_state[PDF_NAV_INPUT_KEY] == 99


# --- (b) 1<=page<=total 클램프 ------------------------------------------------


def test_on_page_change_clamps_high(nav_state: dict[str, object]) -> None:
    """초과 입력은 total로 클램프된다 (TASK 3: 위젯 키는 헬퍼가 덮어쓰지 않음)."""
    nav_state[PDF_NAV_INPUT_KEY] = 99
    viewer_module._on_page_change(10)
    assert _current_page() == 10
    assert nav_state[PDF_NAV_INPUT_KEY] == 99


def test_on_page_change_clamps_low(nav_state: dict[str, object]) -> None:
    """0/음수 입력은 1로 클램프된다 (falsy no-op 금지, 위젯 키 무접촉)."""
    nav_state[PDF_NAV_INPUT_KEY] = 0
    viewer_module._on_page_change(10)
    assert _current_page() == 1
    assert nav_state[PDF_NAV_INPUT_KEY] == 0


def test_on_page_change_pass_through(nav_state: dict[str, object]) -> None:
    """범위 내 입력은 그대로 반영된다."""
    nav_state[PDF_NAV_INPUT_KEY] = 7
    viewer_module._on_page_change(10)
    assert _current_page() == 7


# --- (c) cached total을 prev/next에 전달 (런타임) ------------------------------


def test_prev_clamps_at_first_page(nav_state: dict[str, object]) -> None:
    """1페이지에서 이전 버튼은 1에 머문다 (cached total 사용)."""
    SessionManager.set("current_page", 1, SID)
    viewer_module._on_prev_click(10)
    assert _current_page() == 1


def test_next_clamps_at_last_page(nav_state: dict[str, object]) -> None:
    """마지막 페이지에서 다음 버튼은 total에 머문다 (재조회 없음)."""
    SessionManager.set("current_page", 10, SID)
    viewer_module._on_next_click_callback(10)
    assert _current_page() == 10


def test_next_moves_forward_with_cached_total(
    nav_state: dict[str, object],
) -> None:
    """중간 페이지에서 다음 버튼은 +1 이동한다."""
    viewer_module._on_next_click_callback(10)
    assert _current_page() == 6


# --- (d) fragment deps 축소: 렌더당 1회 resolve, 콜백 재조회 금지 --------------


def test_callbacks_accept_cached_total() -> None:
    """prev/next/page-change 콜백은 렌더가 전달하는 total을 받아야 한다."""
    for name in ("_on_prev_click", "_on_next_click_callback", "_on_page_change"):
        params = inspect.signature(getattr(viewer_module, name)).parameters
        assert "total_pages" in params, f"{name}는 total_pages를 인자로 받아야 함"


def test_next_callback_does_not_reresolve_total() -> None:
    """next 콜백은 _get_pdf_total_pages를 직접 호출하지 않는다 (deps 축소)."""
    source = inspect.getsource(viewer_module._on_next_click_callback)
    assert "_get_pdf_total_pages" not in source


def test_render_resolves_pdf_state_once() -> None:
    """render당 resolve는 _resolve_pdf_state 1회뿐 (직접 페이지수 조회 금지)."""
    source = inspect.getsource(viewer_module.render_pdf_area)
    assert source.count("_resolve_pdf_state()") == 1
    assert "_get_pdf_total_pages" not in source


def test_controls_wire_cached_total_via_args() -> None:
    """컨트롤은 resolved total을 on_click/on_change args로 전달해야 한다."""
    source = inspect.getsource(viewer_module.render_pdf_controls)
    assert source.count("args=(total_pages,)") >= 3, (
        "prev 버튼/next 버튼/number_input 모두 cached total을 args로 전달해야 함"
    )


# --- (e) st.rerun 무관 회귀 가드 -----------------------------------------------


def test_viewer_has_no_st_rerun() -> None:
    """viewer.py에 st.rerun() 실제 호출 금지 (docstring 언급은 허용)."""
    import ast
    import pathlib

    module_file = viewer_module.__file__
    assert module_file is not None
    tree = ast.parse(pathlib.Path(module_file).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rerun"
    ]
    assert not calls
