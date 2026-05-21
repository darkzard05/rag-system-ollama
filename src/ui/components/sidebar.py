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

        # [안정성] 필터링 시 None 에러 방지
        raw_models = [m for m in safe_models if m and "---" not in str(m)]
        embed_keywords = ["embed", "bge", "nomic", "mxbai", "snowflake"]

        embedding_candidates = [
            m for m in raw_models if any(kw in str(m).lower() for kw in embed_keywords)
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

    # 시스템 건강 상태 (실시간 업데이트)
    render_system_health()


@st.fragment(run_every="5s")
def render_system_health():
    """시스템 건강 상태 표시 (CPU/Memory) - 현대화된 커스텀 디자인"""
    import psutil

    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent

    # CPU 부하에 따른 색상 결정
    cpu_color = "#007bff"
    if cpu > 80:
        cpu_color = "#dc3545"  # Error color
    elif cpu > 50:
        cpu_color = "#ffc107"  # Warning color

    st.markdown(
        f"""
        <div class="status-container">
            <div class="status-header">
                <span class="status-title">System Health</span>
                <div class="live-indicator">
                    <div class="live-dot"></div>
                    LIVE
                </div>
            </div>
            <div class="metric-row">
                <!-- CPU Metric -->
                <div class="metric-item">
                    <div class="metric-label-row">
                        <span>CPU Usage</span>
                        <span style="font-weight:700;">{cpu}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {cpu}%; background: {cpu_color};"></div>
                    </div>
                </div>
                <!-- RAM Metric -->
                <div class="metric-item">
                    <div class="metric-label-row">
                        <span>Memory Usage</span>
                        <span style="font-weight:700;">{mem}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {mem}%; background: #6c5ce7;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
