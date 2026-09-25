"""TASK 5 hygiene guards (TDD failing-first): tab-order / label / overflow / i18n / onboarding.

- (a) stream-DOM 제약: 입력창이 메시지 영역보다 먼저 DOM에 있어야 함
  (ui.render_main_content 호출 순서) + 시각 순서는 CSS order:1 + 키보드
  focus-visible 링 유지.
- (b) viewer 페이지 입력: visible label + help (i18n 키 경유).
- (c) main.css overflow-x hidden/clip + 모바일 단일 fixed + order:1 유지.
- (d) 하드코딩 리터럴 t() 전환 + strings.LANG 단일 토글 배선.
- (e) 온보딩 단일 소스 (chat_build._render_guidance_panel 게이트 일원화).
- (f) 1.60 버전가드 + 새 !important/testID 셀렉터 0건 (베이스라인 고정).
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

import ui.components.chat as chat_module
import ui.components.chat_build as chat_build_module
import ui.components.common as common_module
import ui.components.sidebar as sidebar_module
import ui.components.viewer as viewer_module
import ui.ui as ui_module
from ui.strings import UI_STRINGS, set_lang, t

_VIEWER_FILE = viewer_module.__file__
assert _VIEWER_FILE is not None
_SRC_ROOT = pathlib.Path(_VIEWER_FILE).parent.parent.parent
_CSS_PATH = _SRC_ROOT / "ui" / "styles" / "main.css"


def _css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


# --- (a) 탭순서 / stream-DOM 제약 -------------------------------------------


def test_stream_dom_order_kept_input_before_messages() -> None:
    """입력창이 스트리밍 루프보다 먼저 DOM에 있어야 한다 (제약 검증 결론)."""
    source = inspect.getsource(ui_module.render_main_content)
    assert "render_chat_input_area" in source
    assert "render_chat_messages_area" in source
    assert source.index("render_chat_input_area") < source.index(
        "render_chat_messages_area"
    ), "input이 스트리밍 중 DOM에 있어야 하므로 DOM 순서 유지 (시각 순서는 order:1)"


def test_input_visual_order_and_focus_ring() -> None:
    """시각적 하단 고정(order:1) + 키보드 focus-visible 링이 있어야 한다."""
    css = _css()
    assert "order: 1" in css, "input-first DOM의 시각 보정(order:1) 유지"
    assert ":focus-visible" in css, "키보드 탭 포커스 링 유지"


# --- (b) viewer 페이지 입력 라벨 --------------------------------------------


def test_viewer_page_input_has_visible_label_and_help() -> None:
    """number_input에 visible label + help (하드코딩 'Page' 금지)."""
    source = inspect.getsource(viewer_module.render_pdf_controls)
    assert 'label_visibility="visible"' in source, "Page 입력에 visible label 필요"
    assert "help=" in source, "Page 입력에 help 필요"
    assert '"Page"' not in source, "하드코딩 라벨은 t() 키로 대체"
    assert "pdf_page_label" in source
    assert "pdf_page_help" in source


# --- (c) CSS overflow / 모바일 단일 fixed ------------------------------------


def test_css_overflow_x_guards() -> None:
    """좁은 화면 가로 넘침 방지: overflow-x clip 가드가 있어야 한다."""
    css = _css()
    assert css.count("overflow-x: clip") >= 2, "word-break 블록 + 헤더에 clip 가드"


def test_mobile_single_fixed_chat_input() -> None:
    """모바일에서 chat input 고정은 1곳만 (데스크탑 sticky와 이중고정 금지)."""
    css = _css()
    assert css.count("position: fixed") == 1, "fixed 이중고정 해소: 1곳만 허용"
    mobile_block = css[css.index("@media (max-width: 768px)") :]
    # fragment 래핑 입력까지 중립화되어야 이중고정이 사라진다 (기존 testID만 재사용).
    assert '> [data-testid="stLayoutWrapper"]' in mobile_block
    assert mobile_block.count("position: static") >= 1


def test_input_visual_order_preserved() -> None:
    """(a) 결론 유지: order:1이 삭제되지 않아야 한다."""
    assert "order: 1" in _css()


# --- (d) i18n -----------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "pdf_page_label",
        "pdf_page_help",
        "onboarding_guide",
        "nav_prev",
        "nav_next",
        "pdf_highlight_load_failed",
        "pdf_error_data",
        "pdf_error_open",
    ],
)
def test_i18n_keys_complete_en_ko(key: str) -> None:
    """신규/재사용 키가 en+ko 테이블에 모두 있어야 한다."""
    assert key in UI_STRINGS["en"], f"en 테이블에 {key} 필요"
    assert key in UI_STRINGS["ko"], f"ko 테이블에 {key} 필요"
    assert UI_STRINGS["en"][key].strip()
    assert UI_STRINGS["ko"][key].strip()


def test_lang_toggle_single_wire() -> None:
    """strings.LANG 토글 하나로 en/ko 전환되어야 한다 (Task4 키 패턴)."""
    import ui.strings as strings_module

    original = strings_module.LANG
    set_lang("ko")
    try:
        assert t("nav_prev") == UI_STRINGS["ko"]["nav_prev"]
    finally:
        set_lang(original)
    assert t("nav_prev") == UI_STRINGS[original]["nav_prev"]


def test_sidebar_lang_toggle_wired() -> None:
    """사이드바에 LANG 단일 토글이 배선되어야 한다."""
    source = inspect.getsource(sidebar_module)
    assert "set_lang" in source, "사이드바에서 strings.LANG 토글로 배선"
    assert "on_language_change" in source, "단일 콜백 경로"


def test_hardcoded_literals_use_t() -> None:
    """viewer/common의 하드코딩 리터럴은 t() 경유여야 한다."""
    viewer_src = inspect.getsource(viewer_module)
    assert 't("nav_prev")' in viewer_src or "t('nav_prev')" in viewer_src
    assert 't("nav_next")' in viewer_src or "t('nav_next')" in viewer_src
    assert "pdf_highlight_load_failed" in viewer_src
    common_src = inspect.getsource(common_module.show_pdf_error)
    assert "pdf_error_data" in common_src
    assert "pdf_error_open" in common_src
    guidance_src = inspect.getsource(chat_build_module._render_guidance_panel)
    assert "onboarding_guide" in guidance_src


# --- (e) 온보딩 단일 소스 -------------------------------------------------------


def test_onboarding_single_source_gate() -> None:
    """온보딩 렌더 판단은 _render_guidance_panel 내부 게이트 단일 소스."""
    timeline_src = inspect.getsource(chat_module._render_unified_timeline)
    # 타임라인 빈 분기는 패널에 위임만 하고 파일 게이트를 중복하지 않는다.
    assert "_render_guidance_panel()" in timeline_src
    branch = timeline_src[timeline_src.index("if not chat_messages") :]
    branch = branch[: branch.index("return") + len("return")]
    assert "last_uploaded_file_name" not in branch, "파일 게이트 중복 금지 (단일 소스)"


def test_onboarding_body_from_strings() -> None:
    """온보딩 본문은 strings 단일 키에서 와야 한다 (산재 리터럴 금지)."""
    guidance_src = inspect.getsource(chat_build_module._render_guidance_panel)
    assert (
        't("onboarding_guide")' in guidance_src
        or "t('onboarding_guide')" in guidance_src
    )


# --- (f) 버전가드 + 새 셀렉터 0건 -----------------------------------------------


def test_css_version_guard_present() -> None:
    """testID CSS에 Streamlit 1.60 DOM 계약 가드가 있어야 한다."""
    css = _css()
    assert "1.60" in css, "1.60 버전가드 주석 필요"
    assert "DOM" in css


def test_no_new_important_or_testid_selectors() -> None:
    """새 !important/testID 셀렉터 0건 (고유 testID 16종 고정 + !=200 초과 금지)."""
    import re

    css = _css()
    assert css.count("!important") <= 200, "새 !important 추가 금지"
    names = set(re.findall(r'data-testid="([^"]+)"', css))
    baseline = {
        "stAppViewContainer",
        "stBottomBlockContainer",
        "stChatInput",
        "stChatMessage",
        "stColumn",
        "stElementContainer",
        "stExpander",
        "stHeader",
        "stHorizontalBlock",
        "stIFrame",
        "stLayoutWrapper",
        "stMain",
        "stMainBlockContainer",
        "stPopoverBody",
        "stSidebarCollapseButton",
        "stVerticalBlock",
    }
    assert names <= baseline, f"새 testID 셀렉터 추가 금지: {names - baseline}"
