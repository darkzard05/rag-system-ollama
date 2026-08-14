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
):
    """Render the settings section logic (accessibility-optimized)."""
    safe_models = available_models if isinstance(available_models, list) else []

    # 1. 문서 업로드 섹션
    with st.container(border=True):
        st.file_uploader(
            "Upload PDF Document",
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
            disabled=is_generating,
        )
        # 업로더 on_change 콜백 이후에 세션에서 다시 읽어 rerun 없이도
        # 새 파일명이 즉시 반영되도록 한다.
        current_file_name = (
            SessionManager.get("last_uploaded_file_name") or current_file_name
        )
        if current_file_name:
            st.caption(f"Current File: :green[{current_file_name}]")

    # 2. 고급 설정 (익스팬더)
    with st.expander("Settings", expanded=False):
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
            "LLM Model Selection",
            actual_llms,
            index=def_idx,
            key="model_selector",
            on_change=model_selector_callback,
            disabled=is_generating or is_swapping_model,
        )

        # 모델 새로고침 (Ollama에서 모델 목록 재조회)
        if (
            st.button(
                "Refresh Models",
                use_container_width=True,
                type="primary",
                help="Refresh the list of models available in Ollama.",
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
            "Embedding Model Selection",
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
                "New Chat",
                use_container_width=True,
                type="primary",
                help="Start a new chat, keeping the uploaded documents.",
                key="new_chat_btn",
                disabled=is_generating or is_building,
            )
            and new_chat_callback
        ):
            new_chat_callback()

        # 초기화 (파괴적 동작 — 확인 다이얼로그 + 생성 중 비활성 + 시각 위계 하향)
        if st.button(
            "Reset All",
            use_container_width=True,
            type="secondary",
            help="Delete all conversations and data.",
            key="reset_btn",
            disabled=is_generating,
        ):
            _confirm_reset_all()


@st.dialog("Confirm Reset All")
def _confirm_reset_all() -> None:
    """Confirmation dialog before Reset All (module-level for st.dialog constraint).

    Since this is destructive, both cancel and confirm paths are provided.
    Cancel does nothing; confirm resets all state.
    """
    st.warning("All conversations and data will be deleted. Continue?")
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button(
            "Cancel",
            use_container_width=True,
            key="reset_confirm_cancel_btn",
        ):
            st.rerun()
    with col_confirm:
        if st.button(
            "Delete",
            use_container_width=True,
            type="primary",
            key="reset_confirm_btn",
        ):
            SessionManager.reset_all_state()
            st.rerun()
