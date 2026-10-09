"""Reference 한 줄 요약 목록 (render_reference_list) 계약 테스트.

- 문서별 최대 1행, 상위 5행 상한, 80자 절단 + ``...`` 접미사 (trailing dot-run 접기).
- 행 라벨 ``"{i}. :blue[{요약}] · p.{N}"``, 키 ``reflink_{msg_id}_{i}_{page}``.
- tertiary + stretch 버튼, 클릭은 ``_handle_page_jump`` on_click 콜백
  (렌더 시점 토큰 세팅 금지 — P0 회귀 가드).
- 발췌 마크다운 정제(헤더/표/강조) + 링크형 CSS 스코프 계약 (R2).
"""

import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import ui.components.chat_references as refs
from ui.components.chat_references import (
    _strip_markdown_for_label,
    render_reference_list,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _doc(page: int, text: str, doc_id: str | None = None) -> MagicMock:
    doc = MagicMock()
    meta: dict = {"page": page}
    if doc_id is not None:
        meta["doc_id"] = doc_id
    doc.metadata = meta
    doc.page_content = text
    return doc


def _render(documents, citations=None, **kwargs):
    st_mock = MagicMock()
    with patch.object(refs, "st", st_mock):
        result = render_reference_list("m1", documents, citations, **kwargs)
    return result, st_mock


def _button_calls(st_mock):
    return st_mock.button.call_args_list


def test_long_excerpt_truncated_single_line() -> None:
    """80자 초과 발췌는 ``...``로 끝나고 줄바꿈이 없다."""
    long_text = "관련 문장입니다. " * 30
    ok, st_mock = _render([_doc(3, long_text)])
    assert ok is True
    (call,) = _button_calls(st_mock)
    label = call.args[0]
    assert label.startswith("1. "), label
    assert "· p.3" in label
    assert "..." in label
    assert "\n" not in label


def test_duplicate_document_single_row() -> None:
    """동일 문서 중복 시 1행만 렌더한다."""
    doc = _doc(2, "중복 문서 본문입니다.")
    ok, st_mock = _render([doc, doc])
    assert ok is True
    assert len(_button_calls(st_mock)) == 1


def test_max_five_rows() -> None:
    """6개 문서 입력 시 상위 5행까지만 렌더한다."""
    docs = [_doc(i, f"문서 {i} 본문입니다.") for i in range(1, 7)]
    ok, st_mock = _render(docs)
    assert ok is True
    calls = _button_calls(st_mock)
    assert len(calls) == 5
    assert calls[0].args[0].startswith("1. ")
    assert calls[4].args[0].startswith("5. ")


def test_citation_span_preferred_over_content() -> None:
    """citation text_span이 있으면 page_content 대신 사용한다."""
    doc = _doc(4, "문서 원문 본문입니다.", doc_id="docA")
    citations = [
        {"doc_id": "docA", "section": "S", "page": 4, "text_span": "인용 핵심 문장"}
    ]
    ok, st_mock = _render([doc], citations)
    assert ok is True
    (call,) = _button_calls(st_mock)
    assert "인용 핵심 문장" in call.args[0]
    assert "문서 원문" not in call.args[0]


def test_citation_page_fallback_to_doc_meta() -> None:
    """인용에 페이지가 없으면 문서 메타 페이지를 사용한다."""
    doc = _doc(9, "메타 페이지 문서입니다.", doc_id="docB")
    citations = [{"doc_id": "docB", "section": "S", "text_span": "핵심"}]
    ok, st_mock = _render([doc], citations)
    assert ok is True
    (call,) = _button_calls(st_mock)
    assert "· p.9" in call.args[0]


def test_button_wiring_tertiary_and_callback() -> None:
    """행 버튼은 tertiary + stretch + _handle_page_jump 콜백이다."""
    ok, st_mock = _render([_doc(5, "배선 확인 문장입니다.")])
    assert ok is True
    (call,) = _button_calls(st_mock)
    assert call.kwargs["type"] == "tertiary"
    assert call.kwargs["width"] == "stretch"
    assert call.kwargs["on_click"] is refs._handle_page_jump
    assert call.kwargs["args"] == (5,)
    assert call.kwargs["key"].startswith("reflink_m1_1_5")


def test_empty_documents_returns_false() -> None:
    """문서/인용이 모두 비면 見出 없이 False."""
    ok, st_mock = _render([])
    assert ok is False
    st_mock.markdown.assert_not_called()
    st_mock.button.assert_not_called()


def test_strip_markdown_headers_and_bullets() -> None:
    """행머리 # 마커와 불릿이 제거된다 (P1)."""
    assert _strip_markdown_for_label("# 3 CM3 Aghajanyan") == "3 CM3 Aghajanyan"
    assert (
        _strip_markdown_for_label("## 4.1 IMAGE MODALITY Although")
        == "4.1 IMAGE MODALITY Although"
    )
    assert _strip_markdown_for_label("- bullet item") == "bullet item"


def test_strip_markdown_table_pipes_and_emphasis() -> None:
    """표 파이프·강조·링크가 걷어내고 snake_case는 보존된다 (P1)."""
    assert "DF-GAN (Tao et al., 2020) 21.42" in _strip_markdown_for_label(
        "|DF-GAN (Tao et al., 2020)|21.42||"
    )
    assert "|" not in _strip_markdown_for_label("|A|B|")
    assert _strip_markdown_for_label("**bold** and `code`") == "bold and code"
    assert (
        _strip_markdown_for_label("[Jukebox paper](https://arxiv.org/abs/x)")
        == "Jukebox paper"
    )
    assert (
        _strip_markdown_for_label("keep pdf_target_page intact")
        == "keep pdf_target_page intact"
    )


def test_row_label_has_no_markdown_markers() -> None:
    """마크다운 원문 문서도 행 라벨에 기호 없이 렌더된다 (P1)."""
    ok, st_mock = _render([_doc(6, "## 4.1.2 IMAGE IN-FILLING Unlike DALL-E, which")])
    assert ok is True
    (call,) = _button_calls(st_mock)
    assert "##" not in call.args[0]
    assert call.args[0].startswith("1. :blue[4.1.2 IMAGE IN-FILLING")


def test_truncation_collapses_trailing_dots() -> None:
    """절단 경계 마침표 런은 단일 ...로 접힌다 (P5)."""
    from ui.components.chat_references import normalize_excerpt

    out = normalize_excerpt("7 RELATED WORK proposed... " + "y" * 200, 80)
    assert "...." not in out
    assert "... ..." not in out
    assert out.endswith("...")


def test_row_label_blue_excerpt() -> None:
    """내용만 파랑(:blue), 번호·페이지는 기본색. 대괄호 잔재는 제거된다."""
    ok, st_mock = _render([_doc(8, "XMC-GAN (Zhao et al.) [draft] 결과입니다.")])
    assert ok is True
    (call,) = _button_calls(st_mock)
    label = call.args[0]
    assert label.startswith("1. :blue[")
    assert "] · p.8" in label
    excerpt_part = label.split(":blue[", 1)[1].rsplit("]", 1)[0]
    assert "[" not in excerpt_part
    assert "]" not in excerpt_part


def test_reference_row_css_contract() -> None:
    """링크형 CSS 계약 (P2/P3/P4): tertiary 스코프 + 밑줄/말줄임, !important 0건."""
    refs_file = refs.__file__
    assert refs_file is not None
    css_path = pathlib.Path(refs_file).parent.parent / "styles" / "main.css"
    css = css_path.read_text(encoding="utf-8")
    scope = '[data-testid="stChatMessage"] [data-testid="stBaseButton-tertiary"]'
    assert scope in css
    for prop in (
        "text-decoration: underline",
        "text-align: left",
        "white-space: nowrap",
        "text-overflow: ellipsis",
    ):
        assert prop in css, f"R2 CSS 속성 누락: {prop}"
    block = css[css.index("/* R2:") :]
    assert "!important" not in block, "R2 블록에 !important 금지"
    # 스코프 전제: tertiary의 유일한 생산자는 render_reference_list이다.
    src_root = pathlib.Path(refs_file).parent.parent.parent
    producers = [
        p
        for p in src_root.rglob("*.py")
        if 'type="tertiary"' in p.read_text(encoding="utf-8")
    ]
    assert [p.name for p in producers] == ["chat_references.py"]


def test_reference_list_click_invokes_handler_via_callback() -> None:
    """P0 회귀: 행 클릭이 on_click 콜백으로 점프 토큰을 정확히 1회 세팅한다."""
    from streamlit.testing.v1 import AppTest

    from core.session import SessionManager

    script_content = """
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from unittest.mock import MagicMock
import ui.components.chat_references as refs
from ui.components.chat_references import render_reference_list
from core.session import SessionManager

doc = MagicMock()
doc.metadata = {"page": 3}
doc.page_content = "클릭 테스트 문장입니다. " * 5

if not getattr(refs._handle_page_jump, "_is_counting_wrapper", False):
    _true_orig_jump = refs._handle_page_jump

    def _counting_jump(p):
        if "__jump_count" not in st.session_state:
            st.session_state["__jump_count"] = 1
        else:
            st.session_state["__jump_count"] = (
                int(st.session_state["__jump_count"]) + 1
            )
        return _true_orig_jump(p)

    _counting_jump._is_counting_wrapper = True  # type: ignore[attr-defined]
    refs._handle_page_jump = _counting_jump

render_reference_list(msg_id="m1", documents=[doc], citations=None)

_token = SessionManager.get("pdf_target_page")
if isinstance(_token, dict) and "page" in _token:
    st.session_state["__jumped_page"] = int(_token["page"])
_cp = SessionManager.get("current_page")
if isinstance(_cp, int):
    st.session_state["__current_page"] = int(_cp)
"""
    SessionManager.reset()
    with open("temp_test_reflist_callback.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    try:
        at = AppTest.from_file("temp_test_reflist_callback.py").run(timeout=60)
        assert not at.exception, at.exception
        assert "__jumped_page" not in at.session_state, (
            "렌더 단계에서 점프 토큰이 세팅됨 (인라인 호출 회귀 의심)"
        )
        btn = next(b for b in at.button if (b.key or "").startswith("reflink_m1_1_"))
        at_after = btn.click().run(timeout=60)
        assert not at_after.exception, at_after.exception
        assert "__jump_count" in at_after.session_state
        assert at_after.session_state["__jump_count"] == 1
        assert at_after.session_state["__jumped_page"] == 3
        assert at_after.session_state["__current_page"] == 3
    finally:
        if os.path.exists("temp_test_reflist_callback.py"):
            os.remove("temp_test_reflist_callback.py")
