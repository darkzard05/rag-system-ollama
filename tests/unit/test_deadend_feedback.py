"""Dead-end feedback repair (TASK 4, TDD failing-first).

수호 조건:
- (a) chat.py stopped dead-end: 하드코딩 캡션 대신 ``t("status_stopped")`` +
  ``get_build_error_actions`` 미러링 retry CTA (on_click 콜백, Task3 계약 유지).
- (b) chat_references.py generating 분기: 클릭 불가 markdown span 대신
  disabled ``st.button`` (동일 키 네임스페이스, rerun/직접쓰기 없음).
- (c) viewer.py 좌표캐시 경고: 렌더당 cause별 1건으로 dedupe (N개 실패 문서도
  ``st.warning`` 1회).
- (d) Task3 계약 유지: 헬퍼 내 st.rerun/위젯키 직접쓰기 금지.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import ui.components.chat as chat_module
import ui.components.chat_references as refs_module
import ui.components.viewer as viewer_module
from ui.strings import UI_STRINGS, set_lang, t


def _parse_source(mod: Any) -> ast.AST:
    src_file = mod.__file__
    assert src_file is not None
    import pathlib

    return ast.parse(pathlib.Path(src_file).read_text(encoding="utf-8"))


def _func_node(tree: ast.AST, name: str) -> Any:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"function {name} not found")


def _has_rerun(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr == "rerun":
                return True
    return False


# ---------------------------------------------------------------------------
# (a) stopped dead-end: t() key + retry CTA
# ---------------------------------------------------------------------------


def test_stopped_caption_uses_t_key() -> None:
    """중단 캡션은 하드코딩 문자열이 아니라 t("status_stopped") 경유."""
    assert "status_stopped" in UI_STRINGS["en"], "en 테이블에 status_stopped 필요"
    assert "status_stopped" in UI_STRINGS["ko"], "ko 테이블에 status_stopped 필요"
    src = inspect.getsource(chat_module._draw_streaming_message)
    assert "status_stopped" in src, "stopped dead-end는 t('status_stopped') 사용"
    assert "답변 생성이 사용자에 의해 중단되었습니다." not in src, (
        "하드코딩 캡션 잔존 금지"
    )


def test_stopped_retry_cta_keys_exist() -> None:
    """retry CTA 라벨/도움말은 en/ko 테이블에 존재."""
    assert t("action_retry_answer") != "action_retry_answer"
    set_lang("ko")
    try:
        assert t("action_retry_answer") != "action_retry_answer"
    finally:
        set_lang("en")


def test_stopped_dead_end_renders_retry_button() -> None:
    """내용 없는 stopped 플레이스홀더 렌더 시 caption + retry 버튼 존재."""
    fake_st = MagicMock()
    fake_st.session_state = {}
    with (
        patch.object(chat_module, "st", fake_st),
        patch.object(chat_module.SessionManager, "get", return_value=False),
    ):
        chat_module._draw_streaming_message({"content": "", "msg_id": "m1"}, "sid1")
    captions = [str(c.args[0]) for c in fake_st.caption.call_args_list]
    assert captions, "stopped dead-end는 caption을 렌더해야 한다"
    assert fake_st.button.called, "stopped dead-end는 retry CTA 버튼을 렌더해야 한다"
    _, kwargs = fake_st.button.call_args
    assert kwargs.get("on_click") is not None, "retry는 on_click 콜백 경유 (Task3)"
    assert kwargs.get("key"), "retry 버튼은 안정 위젯 키 필요"


def test_stopped_retry_action_mirrors_build_error_shape() -> None:
    """retry 액션은 get_build_error_actions 구조를 미러링."""
    actions = chat_module.get_stopped_answer_actions()
    assert actions["retryable"] is True
    assert actions["retry_kind"] == "answer"
    assert actions["cta"], "CTA micro-copy 필요"
    assert actions["message"] == t("status_stopped")


def test_stopped_helper_has_no_rerun_or_widget_writes() -> None:
    """Task3 계약: stopped 헬퍼 내 st.rerun/위젯키 직접쓰기 금지."""
    tree = _parse_source(chat_module)
    for name in ("_draw_streaming_message", "_retry_stopped_answer"):
        node = _func_node(tree, name)
        assert not _has_rerun(node), f"{name}: st.rerun 금지 (Task3)"
    retry_src = inspect.getsource(chat_module._retry_stopped_answer)
    assert "session_state" not in retry_src, (
        "retry 콜백은 st.session_state 직접쓰기 금지"
    )
    assert "MAIN_CHAT_INPUT_KEY" not in retry_src
    assert "PDF_NAV_INPUT_KEY" not in retry_src


# ---------------------------------------------------------------------------
# (b) generating: disabled button (no dead span)
# ---------------------------------------------------------------------------


def test_generating_refs_use_disabled_button() -> None:
    """generating 분기는 span(markdown) 대신 disabled st.button."""
    src = inspect.getsource(refs_module._render_references_content)
    gen_branch = src.split("if generating:")[1].split("else:")[0]
    assert "st.button" in gen_branch, "generating 분기는 st.button 사용"
    assert "disabled=True" in gen_branch.replace(
        "disabled = True", "disabled=True"
    ).replace("disabled", "disabled"), "generating 버튼은 disabled"
    assert "data-doc-id" not in gen_branch, "dead span 잔존 금지"


def test_generating_refs_render_disabled_runtime() -> None:
    """런타임: generating=True 렌더는 disabled 버튼, markdown 미호출."""
    fake_st = MagicMock()
    docs = [{"metadata": {"page": 3}}]
    with patch.object(refs_module, "st", fake_st):
        refs_module._render_references_content(
            "m1", docs, citations=None, generating=True
        )
    assert fake_st.button.called, "generating 렌더는 button이어야 한다"
    _, kwargs = fake_st.button.call_args
    assert kwargs.get("disabled") is True
    assert not fake_st.markdown.called, "generating 렌더는 dead span(markdown) 금지"


# ---------------------------------------------------------------------------
# (c) viewer warnings: one per cause per render
# ---------------------------------------------------------------------------


def _doc_with_coord_error(source: str) -> dict[str, Any]:
    return {"metadata": {"coord_cache_error": True, "file_path": source}}


def test_coord_warnings_deduped_to_one_per_cause() -> None:
    """3개 실패 문서도 st.warning 1회 (cause별 1건/렌더)."""
    docs = [
        _doc_with_coord_error("a.pdf"),
        _doc_with_coord_error("a.pdf"),  # 동일 소스 중복
        _doc_with_coord_error("b.pdf"),
        _doc_with_coord_error("c.pdf"),
    ]
    fake_st = MagicMock()
    with (
        patch.object(viewer_module, "st", fake_st),
        patch.object(viewer_module.SessionManager, "get", return_value=docs),
    ):
        viewer_module._render_coord_cache_warnings()
    assert fake_st.warning.call_count == 1, (
        f"cause별 1건이어야 하나 {fake_st.warning.call_count}회 호출"
    )
