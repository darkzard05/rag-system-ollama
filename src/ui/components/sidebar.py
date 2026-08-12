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
    new_chat_callback=None,
    refresh_models_callback=None,
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
        new_chat_callback,
        refresh_models_callback,
        is_generating,
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
        # 업로더 on_change 콜백 이후에 세션에서 다시 읽어 rerun 없이도
        # 새 파일명이 즉시 반영되도록 한다.
        current_file_name = (
            SessionManager.get("last_uploaded_file_name") or current_file_name
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

        # 모델 새로고침 (Ollama에서 모델 목록 재조회)
        if (
            st.button(
                "↻ 모델 새로고침",
                use_container_width=True,
                type="primary",
                help="Ollama에서 사용 가능한 모델 목록을 다시 불러옵니다.",
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
                "새 대화",
                use_container_width=True,
                type="primary",
                help="대화 내용만 초기화하고 문서는 유지합니다.",
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
            help="모든 대화와 데이터를 초기화합니다.",
            key="reset_btn",
            disabled=is_generating,
        ):
            _confirm_reset_all()


@st.dialog("Reset All 확인")
def _confirm_reset_all() -> None:
    """Reset All 실행 전 확인 다이얼로그 (모듈 레벨 정의 — st.dialog 제약).

    파괴적 동작이므로 취소/확인 두 경로를 제공합니다.
    취소 시 아무것도 수행하지 않고, 확인 시 전체 상태를 초기화합니다.
    """
    st.warning("모든 대화·데이터가 삭제됩니다. 계속할까요?")
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button(
            "취소",
            use_container_width=True,
            key="reset_confirm_cancel_btn",
        ):
            st.rerun()
    with col_confirm:
        if st.button(
            "확인 삭제",
            use_container_width=True,
            type="primary",
            key="reset_confirm_btn",
        ):
            SessionManager.reset_all_state()
            st.rerun()
