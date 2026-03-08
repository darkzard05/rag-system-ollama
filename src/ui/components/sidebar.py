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


def render_sidebar(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    is_generating=False,
    current_file_name=None,
    current_embedding_model=None,
    available_models=None,
):
    """사이드바 최상위 렌더링 함수"""
    pdf_path = SessionManager.get("pdf_file_path")
    is_expanded = bool(pdf_path)

    with st.sidebar:
        # [최적화] CSS 픽셀값과 정확히 일치하는 비율을 설정하여 1열 너비를 고정합니다.
        # 비확장 시 1열(300px)이 전체를 점유하도록 2열을 극소화합니다.
        column_ratios = [300, 700] if is_expanded else [300, 1]
        col_settings, col_viewer = st.columns(column_ratios)

        with col_settings:
            st.markdown(
                "<div class='sidebar-header'>🤖 RAG System</div>",
                unsafe_allow_html=True,
            )
            # 설정창 영역 (CSS 클래스로 높이 제어)
            with st.container():
                _render_settings_internal(
                    file_uploader_callback,
                    model_selector_callback,
                    embedding_selector_callback,
                    is_generating,
                    current_file_name,
                    available_models,
                )

        with col_viewer:
            # 2열은 PDF가 있을 때만 채워지며, 없을 때는 CSS에 의해 너비가 0이 됩니다.
            if is_expanded:
                st.markdown(
                    "<div class='sidebar-header'>📄 문서 미리보기</div>",
                    unsafe_allow_html=True,
                )
                with st.container():
                    from ui.components.viewer import render_pdf_viewer

                    render_pdf_viewer()
            else:
                st.empty()


def _render_settings_internal(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    is_generating,
    current_file_name,
    available_models,
):
    """사이드바의 설정 섹션 실제 렌더링 로직"""
    with st.container(border=True):
        st.subheader("📄 문서 업로드")
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

    with st.container(border=True):
        st.subheader("⚙️ 모델 설정")

        raw_models = [m for m in (available_models or []) if "---" not in m]
        embed_keywords = ["embed", "bge", "nomic", "mxbai", "snowflake"]

        embedding_candidates = [
            m for m in raw_models if any(kw in m.lower() for kw in embed_keywords)
        ]
        actual_embeddings = sorted(
            set(AVAILABLE_EMBEDDING_MODELS + embedding_candidates)
        )
        if DEFAULT_EMBEDDING_MODEL not in actual_embeddings:
            actual_embeddings.append(DEFAULT_EMBEDDING_MODEL)
        actual_embeddings.sort()

        llm_candidates = [m for m in raw_models if m not in embedding_candidates]
        actual_llms = llm_candidates if llm_candidates else [DEFAULT_OLLAMA_MODEL]
        if DEFAULT_OLLAMA_MODEL not in actual_llms:
            actual_llms.append(DEFAULT_OLLAMA_MODEL)
        actual_llms.sort()

        st.write("**💬 sLLM**")
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

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        st.write("**🔍 임베딩 모델**")
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

    with st.expander("🛠️ 고급 설정", expanded=False):
        if st.button("🗑️ VRAM 비우기", use_container_width=True):
            from common.utils import sync_run
            from core.model_loader import ModelManager

            sync_run(ModelManager.clear_vram())
            st.toast("VRAM 정리 완료")

        if st.button(
            "🔄 시스템 전체 초기화",
            use_container_width=True,
            type="primary",
            help="UI가 멈추거나 오류가 발생했을 때 클릭하세요. 모든 대화와 문서가 초기화됩니다.",
        ):
            SessionManager.reset_all_state()
            st.rerun()
