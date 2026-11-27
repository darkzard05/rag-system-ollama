"""
Sidebar settings and management component.
(Accessibility: CSS classes instead of label_visibility="collapsed")
"""

import re
from collections.abc import Callable

import streamlit as st

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ENABLE_VECTOR_CACHE,
    GRADING_ENABLED,
    QUERY_CACHE_ENABLED,
    RERANKER_ENGINE,
)
from core.session import SessionManager
from ui.strings import set_lang, t
from ui.widget_keys import (
    CONFIRM_RESET_PENDING_KEY,
    GRADING_TOGGLE_KEY,
    LANGUAGE_SELECTOR_KEY,
    QUERY_CACHE_TOGGLE_KEY,
    RERANKER_ENGINE_SELECT_KEY,
    SETTINGS_DIALOG_OPEN_KEY,
    VECTOR_CACHE_TOGGLE_KEY,
    reset_all_key,
)

_LANG_OPTIONS: dict[str, str] = {"한국어": "ko", "English": "en"}

# UX-7: parameter-size token in model IDs (e.g. "4b", "70B").
_SIZE_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b", re.IGNORECASE)


def format_llm_option(model_id: str) -> str:
    """UX-7: raw ID + role badge for display (stored value stays raw).

    Size token (``4b``/``70B``) → ``"4B · <badge>"`` suffix; 8B and below
    gets the fast/light badge, above gets high-capability. IDs without a
    size token render raw (no guessing).
    """
    raw = str(model_id)
    match = _SIZE_TOKEN_RE.search(raw)
    if match is None:
        return raw
    try:
        size = float(match.group(1))
    except ValueError:
        return raw
    badge = t("model_badge_fast") if size <= 8 else t("model_badge_standard")
    return f"{raw} · {match.group(1)}B · {badge}"


def resolve_model_option(value: object) -> str:
    """Recover the raw Ollama model id from a selectbox value.

    Streamlit serializes selectbox state through ``format_func``. In an
    ``on_change`` callback the widget key can still hold the formatted label
    (for example ``"qwen3:4b… · 4B · fast"``). Feeding that label to Ollama
    fails with ``invalid model name``. ``format_llm_option`` appends badges
    after ``" · "``, so the id is everything before the first separator.
    """
    return str(value or "").strip().split(" · ", 1)[0]


def on_language_change() -> None:
    """LANG 단일 토글 콜백: 셀렉터 읽기 → 스토어 + strings.LANG에 반영."""
    label = st.session_state.get(LANGUAGE_SELECTOR_KEY, "English")
    code = _LANG_OPTIONS.get(str(label), "en")
    SessionManager.set("ui_lang", code)
    set_lang(code)


def _on_settings_dismiss() -> None:
    """Clear the queued-modal flag when the user closes the settings dialog."""
    st.session_state[SETTINGS_DIALOG_OPEN_KEY] = False


def _request_reset_confirm() -> None:
    """Defer the reset confirmation to app scope (one dialog per script run)."""
    st.session_state[CONFIRM_RESET_PENDING_KEY] = True
    st.session_state[SETTINGS_DIALOG_OPEN_KEY] = False
    st.rerun()


def _on_vector_cache_change() -> None:
    ENABLE_VECTOR_CACHE.set(st.session_state[VECTOR_CACHE_TOGGLE_KEY])


def _on_query_cache_change() -> None:
    QUERY_CACHE_ENABLED.set(st.session_state[QUERY_CACHE_TOGGLE_KEY])


def _on_grading_change() -> None:
    GRADING_ENABLED.set(st.session_state[GRADING_TOGGLE_KEY])


def _on_reranker_engine_change() -> None:
    RERANKER_ENGINE.set(st.session_state[RERANKER_ENGINE_SELECT_KEY])


def _reset_runtime_settings() -> None:
    for setting in (
        ENABLE_VECTOR_CACHE,
        QUERY_CACHE_ENABLED,
        GRADING_ENABLED,
        RERANKER_ENGINE,
    ):
        setting.reset()


def _render_sidebar_logo():
    """Render the brand logo at the top of the sidebar (C8: theme-driven CSS)."""
    st.markdown(
        """
<div class="rag-brand">
    <div class="rag-brand__name">GraphRAG-Ollama</div>
    <div class="rag-brand__sub">Local RAG · PDF Chat</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_upload_section(file_uploader_callback, is_generating: bool) -> None:
    """PDF uploader stays in the sidebar; the rest of the settings move to the modal."""
    is_building = bool(SessionManager.get("is_building_rag", False))
    with st.container(border=True):
        st.file_uploader(
            t("sidebar_upload_label"),
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
            help=t("sidebar_upload_help"),
            disabled=is_generating or is_building,
        )


def render_settings_content(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    new_chat_callback=None,
    refresh_models_callback=None,
    is_generating=False,
    is_swapping_model=False,
    current_file_name=None,
    current_embedding_model=None,
    available_models=None,
    ollama_reachable=True,
):
    """Render the sidebar entry point: brand, uploader, and the settings button."""
    set_lang(str(SessionManager.get("ui_lang", "en") or "en"))
    _render_sidebar_logo()
    # INT-4: locked controls need a reason — chat input stays live as ■ stop,
    # settings wait for the turn to end.
    if is_generating:
        st.caption(t("sidebar_locked_generating"))
    if is_swapping_model:
        st.info(t("sidebar_switching"))
    _render_upload_section(file_uploader_callback, is_generating)
    if st.button(
        t("sidebar_settings"),
        width="stretch",
        type="secondary",
        icon=":material/settings:",
        help=t("sidebar_settings_help"),
        key="settings_dialog_button",
    ):
        st.session_state[SETTINGS_DIALOG_OPEN_KEY] = True


def _render_settings_internal(
    model_selector_callback,
    embedding_selector_callback,
    new_chat_callback,
    refresh_models_callback,
    is_generating,
    is_swapping_model,
    current_file_name,
    available_models,
    ollama_reachable=True,
    reset_callback: Callable[[], None] | None = None,
):
    # LANG 단일 배선: 스토어 값이 매 렌더 strings 활성 언어를 결정한다.
    set_lang(str(SessionManager.get("ui_lang", "en") or "en"))
    """Render the settings section logic (accessibility-optimized)."""
    safe_models = available_models if isinstance(available_models, list) else []
    is_building = bool(SessionManager.get("is_building_rag", False))

    # DEFECT #6: loading indicator while the LLM model is being swapped.
    if is_swapping_model:
        st.info(t("sidebar_switching"))

    st.caption(t("settings_intro"))
    st.space("xsmall")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.caption(t("settings_group_model"))
        st.space("xsmall")
        # UI 언어 토글 (단일 콜백 on_language_change 경유)
        current_lang = str(SessionManager.get("ui_lang", "en") or "en")
        lang_labels = list(_LANG_OPTIONS.keys())
        try:
            lang_idx = list(_LANG_OPTIONS.values()).index(current_lang)
        except ValueError:
            lang_idx = 1
        st.selectbox(
            t("sidebar_language"),
            lang_labels,
            index=lang_idx,
            key=LANGUAGE_SELECTOR_KEY,
            on_change=on_language_change,
            help=t("sidebar_language_help"),
        )
        st.space("xsmall")

        # 모델 설정 그룹
        from core.model_catalog import get_filtered_models

        filtered = get_filtered_models(safe_models)
        actual_llms = filtered["llm"]
        actual_embeddings = filtered["embedding"]

        # LLM 선택
        last_model = SessionManager.get("last_selected_model") or DEFAULT_OLLAMA_MODEL
        if last_model not in actual_llms:
            last_model = actual_llms[0]
        try:
            def_idx = actual_llms.index(last_model)
        except ValueError:
            def_idx = 0

        st.selectbox(
            t("sidebar_llm_label"),
            actual_llms,
            index=def_idx,
            key="model_selector",
            on_change=model_selector_callback,
            format_func=format_llm_option,
            help=t("sidebar_llm_help"),
            disabled=is_generating or is_swapping_model or not ollama_reachable,
        )
        st.space("xsmall")

        if not ollama_reachable:
            st.caption(t("sidebar_ollama_hint"))

        # 모델 새로고침 (Ollama에서 모델 목록 재조회 — 보조 동작이므로 하향)
        if (
            st.button(
                t("sidebar_refresh"),
                width="content",
                type="secondary",
                help=t("sidebar_refresh_help"),
                key="refresh_models_btn",
                disabled=is_generating,
            )
            and refresh_models_callback
        ):
            refresh_models_callback()
        st.space("xsmall")

        # 임베딩 선택
        current_emb = (
            SessionManager.get("last_selected_embedding_model")
            or DEFAULT_EMBEDDING_MODEL
        )
        if current_emb not in actual_embeddings:
            current_emb = actual_embeddings[0]
        try:
            emb_idx = actual_embeddings.index(current_emb)
        except ValueError:
            emb_idx = 0

        st.selectbox(
            t("sidebar_embedding_label"),
            actual_embeddings,
            index=emb_idx,
            key="embedding_model_selector",
            on_change=embedding_selector_callback,
            help=t("sidebar_embedding_help"),
            disabled=is_generating or is_swapping_model or not ollama_reachable,
        )
        st.space("xsmall")

    with right:
        st.caption(t("settings_group_behavior"))
        st.space("xsmall")
        st.toggle(
            t("setting_vector_cache"),
            value=bool(ENABLE_VECTOR_CACHE.get()),
            key=VECTOR_CACHE_TOGGLE_KEY,
            on_change=_on_vector_cache_change,
            help=t("setting_vector_cache_help"),
        )
        st.space("xsmall")
        st.toggle(
            t("setting_query_cache"),
            value=bool(QUERY_CACHE_ENABLED.get()),
            key=QUERY_CACHE_TOGGLE_KEY,
            on_change=_on_query_cache_change,
            help=t("setting_query_cache_help"),
        )
        st.space("xsmall")
        engine_options = ["auto", "flashrank", "semantic"]
        engine = str(RERANKER_ENGINE.get())
        st.selectbox(
            t("setting_reranker_engine"),
            engine_options,
            index=engine_options.index(engine) if engine in engine_options else 0,
            key=RERANKER_ENGINE_SELECT_KEY,
            on_change=_on_reranker_engine_change,
            help=t("setting_reranker_engine_help"),
        )
        st.space("xsmall")
        st.toggle(
            t("setting_grading"),
            value=bool(GRADING_ENABLED.get()),
            key=GRADING_TOGGLE_KEY,
            on_change=_on_grading_change,
            help=t("setting_grading_help"),
        )

    # 실행 동작은 본문 아래 별도 바에 모아 설정과 위계를 분리한다.
    st.space("small")
    st.divider()
    st.space("xsmall")
    sid = SessionManager.get_session_id()
    with st.container(horizontal=True, horizontal_alignment="right"):
        if (
            st.button(
                t("sidebar_new_chat"),
                width="content",
                type="primary",
                help=t("sidebar_new_chat_help"),
                key="new_chat_btn",
                disabled=is_generating or is_building,
            )
            and new_chat_callback
        ):
            new_chat_callback()
        if st.button(
            t("sidebar_reset"),
            width="content",
            type="secondary",
            help=t("sidebar_reset_help"),
            key=reset_all_key(sid),
            disabled=is_generating or is_building,
        ):
            if reset_callback:
                reset_callback()
            else:
                _confirm_reset_all()


def _render_settings_modal(
    model_selector_callback,
    embedding_selector_callback,
    new_chat_callback,
    refresh_models_callback,
    is_generating: bool,
    is_swapping_model: bool,
    current_file_name,
    available_models,
    ollama_reachable: bool,
) -> None:
    """Body of the compact centered settings dialog."""
    if is_generating:
        st.caption(t("sidebar_locked_generating"))
    # Refresh Models writes this key; read it here so a dialog-fragment rerun
    # sees the new list without waiting for the full app to rerun.
    models = st.session_state.get("available_models_list") or available_models
    _render_settings_internal(
        model_selector_callback,
        embedding_selector_callback,
        new_chat_callback,
        refresh_models_callback,
        is_generating,
        is_swapping_model,
        current_file_name,
        models,
        ollama_reachable,
        reset_callback=_request_reset_confirm,
    )


def render_pending_dialogs(
    model_selector_callback,
    embedding_selector_callback,
    new_chat_callback,
    refresh_models_callback,
    is_generating: bool = False,
    is_swapping_model: bool = False,
    current_file_name=None,
    available_models=None,
    ollama_reachable: bool = True,
) -> None:
    """Render at most one queued modal from the app scope.

    Streamlit allows only one dialog per script run. The reset confirmation
    wins over the settings modal; the settings dialog is created here, outside
    ``st.sidebar``, so it renders centered over the main app.
    """
    if st.session_state.pop(CONFIRM_RESET_PENDING_KEY, False):
        st.session_state[SETTINGS_DIALOG_OPEN_KEY] = False
        _confirm_reset_all()
        return
    if not st.session_state.get(SETTINGS_DIALOG_OPEN_KEY, False):
        return
    dialog = st.dialog(
        t("sidebar_settings"),
        width="medium",
        position="center",
        on_dismiss=_on_settings_dismiss,
    )
    dialog(_render_settings_modal)(
        model_selector_callback,
        embedding_selector_callback,
        new_chat_callback,
        refresh_models_callback,
        is_generating,
        is_swapping_model,
        current_file_name,
        available_models,
        ollama_reachable,
    )


@st.dialog(t("sidebar_reset_title"))
def _confirm_reset_all() -> None:
    """파괴적 동작 전 사용자 확인 모달"""
    st.warning(t("sidebar_reset_body"))
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        # @st.dialog 내부 버튼 클릭은 다이얼로그 프래그먼트만 재실행하므로
        # 모달을 닫기 위해 앱 스코프 st.rerun()을 명시 호출한다.
        if st.button(
            t("sidebar_cancel"),
            width="stretch",
            key="reset_confirm_cancel_btn",
        ):
            st.rerun()
    with col_confirm:
        # SessionManager 초기화 후 모달 종료 및 UI 전체 상태(UIBridge) 동기화를 위해
        # 반드시 앱 스코프 st.rerun()을 호출한다.
        if st.button(
            t("sidebar_confirm_reset"),
            width="stretch",
            type="primary",
            key="reset_confirm_btn",
        ):
            SessionManager.reset_all_state()
            _reset_runtime_settings()
            # 위젯 키는 스토어 초기화의 영향을 받지 않는다. 남겨두면 언어·PDF·모델
            # 셀렉터가 이전 값을 계속 표시하므로 세션 상태까지 비운 뒤 재마운트한다.
            st.session_state.clear()
            st.rerun()
