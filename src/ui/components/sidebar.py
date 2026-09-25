"""
Sidebar settings and management component.
(Accessibility: CSS classes instead of label_visibility="collapsed")
"""

import streamlit as st

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from core.session import SessionManager
from ui.strings import set_lang, t
from ui.widget_keys import LANGUAGE_SELECTOR_KEY, reset_all_key

_LANG_OPTIONS: dict[str, str] = {"한국어": "ko", "English": "en"}


def on_language_change() -> None:
    """LANG 단일 토글 콜백: 셀렉터 읽기 → 스토어 + strings.LANG에 반영."""
    label = st.session_state.get(LANGUAGE_SELECTOR_KEY, "English")
    code = _LANG_OPTIONS.get(str(label), "en")
    SessionManager.set("ui_lang", code)
    set_lang(code)


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
    """Render the settings content (callable outside the sidebar)."""
    _render_sidebar_logo()
    _render_settings_internal(
        file_uploader_callback,
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


def _render_settings_internal(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    new_chat_callback,
    refresh_models_callback,
    is_generating,
    is_swapping_model,
    current_file_name,
    available_models,
    ollama_reachable=True,
):
    # LANG 단일 배선: 스토어 값이 매 렌더 strings 활성 언어를 결정한다.
    set_lang(str(SessionManager.get("ui_lang", "en") or "en"))
    """Render the settings section logic (accessibility-optimized)."""
    safe_models = available_models if isinstance(available_models, list) else []

    # DEFECT #6: loading indicator while the LLM model is being swapped.
    if is_swapping_model:
        st.info(t("sidebar_switching"))

    # 1. 문서 업로드 섹션
    with st.container(border=True):
        st.file_uploader(
            t("sidebar_upload_label"),
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
            disabled=is_generating,
        )

    # 2. 고급 설정 (익스팬더)
    with st.expander(t("sidebar_settings"), expanded=False):
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
        # 모델 설정 그룹
        from core.model_loader import ModelManager

        filtered = ModelManager.get_filtered_models(safe_models)
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
            disabled=is_generating or is_swapping_model or not ollama_reachable,
        )

        if not ollama_reachable:
            st.caption(t("sidebar_ollama_hint"))

        # 모델 새로고침 (Ollama에서 모델 목록 재조회 — 보조 동작이므로 하향)
        if (
            st.button(
                t("sidebar_refresh"),
                use_container_width=True,
                type="secondary",
                help=t("sidebar_refresh_help"),
                key="refresh_models_btn",
                disabled=is_generating,
            )
            and refresh_models_callback
        ):
            refresh_models_callback()

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
            disabled=is_generating or (available_models is None),
        )

        # 새 대화 (문서 유지, 대화만 초기화)
        is_building = bool(SessionManager.get("is_building_rag", False))
        if (
            st.button(
                t("sidebar_new_chat"),
                use_container_width=True,
                type="primary",
                help=t("sidebar_new_chat_help"),
                key="new_chat_btn",
                disabled=is_generating or is_building,
            )
            and new_chat_callback
        ):
            new_chat_callback()

        # 초기화 (파괴적 동작 — 확인 다이얼로그 + 생성 중 비활성 + 시각 위계 하향)
        sid = SessionManager.get_session_id()
        st.divider()
        if st.button(
            t("sidebar_reset"),
            use_container_width=True,
            type="secondary",
            help=t("sidebar_reset_help"),
            key=reset_all_key(sid),
            disabled=is_generating,
        ):
            _confirm_reset_all()


@st.dialog(t("sidebar_reset_title"))
def _confirm_reset_all() -> None:
    """파괴적 동작 전 사용자 확인 모달"""
    st.warning(t("sidebar_reset_body"))
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        # 버튼 위젯 interaction 자체의 리런으로 다이얼로그가 닫히므로 명시 rerun 금지.
        if st.button(
            t("sidebar_cancel"),
            use_container_width=True,
            key="reset_confirm_cancel_btn",
        ):
            pass
    with col_confirm:
        if st.button(
            t("sidebar_confirm_reset"),
            use_container_width=True,
            type="primary",
            key="reset_confirm_btn",
            # 버튼 위젯 interaction 자체의 리런으로 전체 UI가 갱신되므로 명시 rerun 금지.
        ):
            SessionManager.reset_all_state()
