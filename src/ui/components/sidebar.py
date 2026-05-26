"""
사이드바 설정 및 관리 컴포넌트 (고정 2열 레이아웃).
"""

import streamlit as st

from common.config import (
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from core.session import SessionManager


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
    # [수정] 설정창 전용으로 로고는 생략하고 내부 설정만 렌더링하도록 함
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
    """사이드바의 설정 섹션 실제 렌더링 로직 (Mockup과 1:1 일치화)"""
    safe_models = available_models if isinstance(available_models, list) else []

    # 1. 문서 업로드 섹션
    st.markdown(
        '<span class="settings-label">📄 Document Assets</span>', unsafe_allow_html=True
    )
    with st.container(border=True):
        st.file_uploader(
            "PDF 파일 업로드",
            type="pdf",
            key="pdf_uploader",
            on_change=file_uploader_callback,
            disabled=is_generating,
            label_visibility="collapsed",
        )
        if current_file_name:
            st.caption(f"현재 파일: :green[{current_file_name}]")

    # 2. 고급 설정 (익스팬더)
    with st.expander("🛠️ Advanced Configuration", expanded=False):
        # 모델 설정 그룹
        st.markdown(
            '<span class="settings-label" style="margin-top:2px;">⚙️ Model Setup</span>',
            unsafe_allow_html=True,
        )

        # [안정성] ModelManager를 통한 필터링 로직 통합
        from core.model_loader import ModelManager
        filtered = ModelManager.get_filtered_models(safe_models)
        actual_llms = filtered["llm"]
        actual_embeddings = filtered["embedding"]

        # LLM 선택
        st.markdown(
            '<div class="settings-sublabel">Reasoning Engine (sLLM)</div>',
            unsafe_allow_html=True,
        )
        last_model = SessionManager.get("last_selected_model") or DEFAULT_OLLAMA_MODEL
        if last_model not in actual_llms:
            last_model = actual_llms[0]
        try:
            def_idx = actual_llms.index(last_model)
        except ValueError:
            def_idx = 0

        st.selectbox(
            "LLM 선택",
            actual_llms,
            index=def_idx,
            key="model_selector",
            on_change=model_selector_callback,
            disabled=is_generating,
            label_visibility="collapsed",
        )

        # 임베딩 선택
        st.markdown(
            '<div class="settings-sublabel">Embedding Model</div>',
            unsafe_allow_html=True,
        )
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
            "임베딩 선택",
            actual_embeddings,
            index=emb_idx,
            key="embedding_model_selector",
            on_change=embedding_selector_callback,
            disabled=is_generating or (available_models is None),
            label_visibility="collapsed",
        )

        st.markdown(
            "<hr style='margin: 20px 0; opacity: 0.1;'>", unsafe_allow_html=True
        )

        # 시스템 도구 그룹
        st.markdown(
            '<span class="settings-label">🔧 Maintenance</span>', unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)

        # VRAM 비우기 (Secondary)
        col1.button(
            "Clear VRAM",
            use_container_width=True,
            help="GPU 메모리를 비웁니다.",
            key="vram_btn",
            type="secondary",
        )

        # 초기화 (Primary)
        if col2.button(
            "Reset All",
            use_container_width=True,
            type="primary",
            help="모든 대화와 데이터를 초기화합니다.",
            key="reset_btn",
        ):
            SessionManager.reset_all_state()
            st.rerun()

        # VRAM 비우기 로직 (type="secondary" 위젯 클릭 시)
        if st.session_state.get("vram_btn"):
            from common.utils import sync_run
            from core.model_loader import ModelManager

            sync_run(ModelManager.clear_vram())
            st.toast("VRAM 정리 완료")

