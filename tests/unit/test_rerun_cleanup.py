"""Rerun-family cleanup guard (TASK 2, TDD failing-first).

수호 조건:
- 콜백-phase ``st.rerun()`` 0건: ``_cancel_rebuild``/``_on_sample_question_click``
  (chat_build), ``on_new_chat``/``on_refresh_models`` (main),
  ``_confirm_reset_all`` (sidebar). Streamlit 자동 post-callback/위젯-interaction
  리런에 의존하므로 명시 호출 금지.
- 빌드 진행은 단일 ``st.status`` 인플레이스 렌더:
  ``_render_build_progress_block`` 내 ``st.status`` 1회, sleep/while 수동 리런
  루프 금지 (Progress 리런폭풍 → st.status 교체).
- stream-DOM 제약 유지: ``render_main_content``는 입력 영역을 메시지 영역보다
  먼저 렌더하고, submit 핸들러(``render_chat_input_area``)는 ``st.rerun()``을
  호출하지 않는다 (제출-스트리밍 동일 run, ■ 중지 버튼 렌더 보장).
"""

import ast
import inspect
import pathlib
from types import ModuleType

import ui.components.chat as chat_module
import ui.components.chat_build as chat_build_module
import ui.components.sidebar as sidebar_module
import ui.ui as ui_module
from ui.components.chat_build import _render_build_progress_block

_CHAT_FILE = chat_module.__file__
assert _CHAT_FILE is not None
_MAIN_FILE = pathlib.Path(_CHAT_FILE).parent.parent.parent / "main.py"


def _rerun_calls_in(func_name: str, tree: ast.Module) -> list[ast.Call]:
    """주어진 함수 정의 내부의 ``st.rerun()`` 호출 목록을 반환한다."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == func_name
        ):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "rerun"
                ):
                    found.append(child)
    return found


def _parse(module: ModuleType) -> ast.Module:
    module_file = module.__file__
    assert module_file is not None
    return ast.parse(pathlib.Path(module_file).read_text(encoding="utf-8"))


# --- (a) 콜백-phase st.rerun() 0건 -------------------------------------------


def test_no_callback_rerun_in_chat_build() -> None:
    """chat_build 콜백(_cancel_rebuild/_on_sample_question_click)은 rerun 금지."""
    tree = _parse(chat_build_module)
    offenders = {
        name: _rerun_calls_in(name, tree)
        for name in ("_cancel_rebuild", "_on_sample_question_click")
    }
    bad = {name: len(calls) for name, calls in offenders.items() if calls}
    assert not bad, f"콜백 내 st.rerun() 잔존: {bad}"


def test_no_callback_rerun_in_main() -> None:
    """main 콜백(on_new_chat/on_refresh_models)은 rerun 금지."""
    tree = ast.parse(_MAIN_FILE.read_text(encoding="utf-8"))
    offenders = {
        name: _rerun_calls_in(name, tree)
        for name in ("on_new_chat", "on_refresh_models")
    }
    bad = {name: len(calls) for name, calls in offenders.items() if calls}
    assert not bad, f"콜백 내 st.rerun() 잔존: {bad}"


def test_no_callback_rerun_in_sidebar_dialog() -> None:
    """sidebar 리셋 다이얼로그(_confirm_reset_all)는 rerun 금지."""
    tree = _parse(sidebar_module)
    calls = _rerun_calls_in("_confirm_reset_all", tree)
    assert not calls, f"_confirm_reset_all 내 st.rerun() {len(calls)}건 잔존"


# --- (b) 단일 st.status 인플레이스 렌더 ---------------------------------------


def test_build_progress_single_status_no_manual_loop() -> None:
    """빌드 진행 블록은 st.status 1회 + 수동 리런 루프 금지."""
    source = inspect.getsource(_render_build_progress_block)
    tree = ast.parse(source)
    status_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "status"
    ]
    assert len(status_calls) == 1, "st.status는 단독 1회만 사용"
    assert "time.sleep" not in source, "수동 폴링 sleep 금지"
    assert "st.rerun" not in source, "진행 블록 내 수동 rerun 금지"
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.While, ast.For))]
    assert not loops, "Progress 수동 리런 루프 금지 (단일 st.status 인플레이스)"


# --- (c) stream-DOM 제약 유지 --------------------------------------------------


def test_input_renders_before_messages_area() -> None:
    """ui.render_main_content: 입력창이 메시지 영역보다 먼저 렌더되어야 한다."""
    source = inspect.getsource(ui_module.render_main_content)
    tree = ast.parse(source)
    order: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("render_chat_input_area", "render_chat_messages_area")
        ):
            order.append((node.lineno, node.func.id))
    order.sort()
    names = [name for _, name in order]
    assert names == ["render_chat_input_area", "render_chat_messages_area"], (
        f"입력창이 스트리밍 루프보다 먼저 DOM에 있어야 함: {names}"
    )


def test_submit_handler_does_not_split_run() -> None:
    """submit 핸들러(render_chat_input_area)는 st.rerun() 호출 금지."""
    tree = _parse(chat_module)
    calls = _rerun_calls_in("render_chat_input_area", tree)
    assert not calls, "submit-스트리밍 분할 rerun 금지"
