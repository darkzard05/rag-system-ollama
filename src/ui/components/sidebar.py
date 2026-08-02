"""
사이드바 설정 및 관리 컴포넌트.
(접근성 개선: label_visibility="collapsed" 대신 CSS 클래스 사용)
"""

import streamlit as st

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from core.session import SessionManager


def _render_sidebar_logo():
    """사이드바 상단에 브랜드 로고를 렌더링합니다."""
    st.markdown(
        """
<div style="text-align: center; padding: 10px 0; margin-bottom: 10px;">
    <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-color);">
        GraphRAG-Ollama
    </div>
    <div style="font-size: 0.9rem; color: var(--primary-color); opacity: 0.8;">
        Local RAG · PDF Chat
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_settings_content(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    is_generating=False,
    current_file_name=None,
    current_embedding_model=None,
    available_models=None,
):
    """설정창 내부 콘텐츠 렌더링 (사이드바 외부 호출용)"""
    _render_sidebar_logo()
    _render_settings_internal(
        file_uploader_callback,
        model_selector_callback,
        embedding_selector_callback,
        is_generating,
        current_file_name,
        available_models,
    )


def _render_settings_internal(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    is_generating,
    current_file_name,
    available_models,
):
    """사이드바의 설정 섹션 실제 렌더링 로직 (접근성 최적화 버전)"""
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
        if current_file_name:
            st.caption(f"Current File: :green[{current_file_name}]")

    # 2. 고급 설정 (익스팬더)
    with st.expander("🛠️ Settings", expanded=False):
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
            disabled=is_generating,
        )

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

        # 초기화
        if st.button(
            "Reset All",
            use_container_width=True,
            type="primary",
            help="모든 대화와 데이터를 초기화합니다.",
            key="reset_btn",
        ):
            SessionManager.reset_all_state()
            st.rerun()
