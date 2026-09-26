"""Widget-key single navigate_to_page() path guard (TASK 3, TDD failing-first).

수호 조건 (citation-v2 page-sync 개정: 점프 후 Page number_input 동기를 위해
아래 두 곳의 위젯 키 쓰기만 허용한다):
- (a) src 전역에서 INTERACTIVE_KEYS 멤버에 대한 스크립트측 쓰기는 허용 목록에
  든 2건뿐이다: ``common.navigate_to_page``의 콜백 단계 동기화와
  ``viewer._resolve_pdf_state``의 토큰 소비 동기화. 그 외 직접 대입은 금지이며,
  단일 ``navigate_to_page()`` 헬퍼가 SessionManager 키 + 네비 입력 키를 쓴다.
- (b) ``navigate_to_page(1)`` 새파일 리셋: current_page=1 + manual_nav_ts
  갱신 + 네비 입력 키도 1로 동기화.
- (c) Task1 클램프 유지: 헬퍼 하한(>=1) + 호출자 상한은 기존 테스트가 수호.
- (d) Task2 무-콜백리런 유지: 헬퍼 내 st.rerun 추가 금지.
"""

import ast
import inspect
import pathlib
import time

import pytest

import ui.components.common as common_module
from core.session import SessionManager
from ui.widget_keys import INTERACTIVE_KEYS

SID = "test_widget_key_contract"

_common_file = common_module.__file__
assert _common_file is not None
_SRC_ROOT = pathlib.Path(_common_file).parent.parent.parent

# 위젯 키 상수 별칭: 스크립트측 쓰기 검사에서 문자열 리터럴과 동일 취급.
_WIDGET_KEY_ALIASES = {"PDF_NAV_INPUT_KEY", "MAIN_CHAT_INPUT_KEY"}

# Page-sync 개정에서 허용하는 유일한 위젯 키 쓰기 2건 (콜백 단계/토큰 소비
# 동기화 — number_input desync 회귀 방지). 그 외 쓰기는 모두 금지.
_ALLOWED_WIDGET_WRITES = {
    "common.py: subscript-write",
    "viewer.py: subscript-write",
}


def _iter_src_files() -> list[pathlib.Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _widget_write_offenders() -> list[str]:
    """INTERACTIVE_KEYS 멤버에 대한 대입/삭제를 모두 수집한다.

    - ``st.session_state[<widget>] = ...`` (Subscript Store)
    - ``st.session_state.pop(<widget>, ...)`` / ``del st.session_state[<widget>]``
    - ``SessionManager.set(<widget>, ...)`` (동기화 어댑터가 위젯 키에
      그대로 미러링하므로 동일하게 금지)
    읽기(Get)는 허용 — 위젯 상태 읽기는 「default를 매니저에서 받기」 전
    사용자 입력을 확인하는 합법 경로다.
    """
    offenders: list[str] = []

    def _is_widget_key(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and node.value in INTERACTIVE_KEYS:
            return True
        if isinstance(node, ast.Name) and node.id in _WIDGET_KEY_ALIASES:
            return True
        return False

    for path in _iter_src_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # st.session_state[<widget>] = ... (쓰기만, 읽기는 제외)
            if isinstance(node, ast.Subscript) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                target = node.value
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "session_state"
                    and _is_widget_key(node.slice)
                ):
                    offenders.append(f"{path.name}: subscript-write")
            # st.session_state.pop(<widget>, ...) / .update(...) 등
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                recv = node.func.value
                if (
                    isinstance(recv, ast.Attribute)
                    and recv.attr == "session_state"
                    and node.func.attr in ("pop", "update", "clear")
                    and any(_is_widget_key(a) for a in node.args)
                ):
                    offenders.append(f"{path.name}: session_state.{node.func.attr}")
            # SessionManager.set(<widget>, ...) — store 경유 간접 쓰기 금지
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "SessionManager"
                    and node.func.attr == "set"
                    and node.args
                    and _is_widget_key(node.args[0])
                ):
                    offenders.append(f"{path.name}: SessionManager.set")
    return offenders


def test_no_script_side_interactive_key_writes() -> None:
    """허용 목록에 든 page-sync 동기화 2건 외의 위젯 키 쓰기는 0건."""
    offenders = _widget_write_offenders()
    assert set(offenders) <= _ALLOWED_WIDGET_WRITES, (
        f"허용 외 위젯 키 직접 쓰기 잔존: {offenders}"
    )
    assert len(offenders) == len(_ALLOWED_WIDGET_WRITES), (
        f"page-sync 동기화 경로 유실: {offenders}"
    )


def test_main_new_file_path_uses_single_helper() -> None:
    """새파일 전환(main.py)은 navigate_to_page(1) 단일 경로만 사용한다."""
    main_src = (_SRC_ROOT / "main.py").read_text(encoding="utf-8")
    assert "navigate_to_page(1)" in main_src, "새파일 리셋은 navigate_to_page(1) 경유"
    assert "pdf_nav_input_v6" not in main_src, "main.py 위젯 키 직접 참조 금지"


def test_navigate_to_page_is_manager_only() -> None:
    """헬퍼는 매니저 키 + 네비 입력 키를 쓰고 st.rerun은 건드리지 않는다."""
    source = inspect.getsource(common_module.navigate_to_page)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "rerun"
        ):
            raise AssertionError("Task2: 헬퍼 내 st.rerun 추가 금지")
    assert "SessionManager.set" in source
    assert "PDF_NAV_INPUT_KEY" in source


@pytest.fixture
def contract_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    fake_session_state: dict[str, object] = {}
    monkeypatch.setattr(common_module.st, "session_state", fake_session_state)
    return fake_session_state


def test_navigate_to_page_sets_manager_not_widget(
    contract_state: dict[str, object],
) -> None:
    """navigate_to_page(1): current_page=1 + 네비 입력 키도 1로 동기화."""
    before = time.time()
    common_module.navigate_to_page(1)
    assert int(SessionManager.get("current_page", 0)) == 1
    assert float(SessionManager.get("manual_nav_ts", 0)) >= before
    assert contract_state == {"pdf_nav_input_v6": 1}


def test_navigate_to_page_new_file_reset(
    contract_state: dict[str, object],
) -> None:
    """새파일 리셋 시나리오: 이전 페이지 잔류 + 오래된 위젯 값도 1로 동기화."""
    SessionManager.set("current_page", 7, SID)
    contract_state["pdf_nav_input_v6"] = 7  # 이전 문서의 sticky 위젯 값
    common_module.navigate_to_page(1)
    assert int(SessionManager.get("current_page", 0)) == 1
    # sticky 위젯 값도 헬퍼가 1로 맞춘다 (number_input desync 방지).
    assert contract_state["pdf_nav_input_v6"] == 1


def test_navigate_to_page_clamps_low(
    contract_state: dict[str, object],
) -> None:
    """Task1 의미 유지: 0/음수는 1로 클램프 (상한은 호출자 total 클램프가 수호)."""
    common_module.navigate_to_page(0)
    assert int(SessionManager.get("current_page", 0)) == 1
    common_module.navigate_to_page(-4)
    assert int(SessionManager.get("current_page", 0)) == 1
    assert contract_state == {"pdf_nav_input_v6": 1}
