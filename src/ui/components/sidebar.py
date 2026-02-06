"""
사이드바 설정 및 관리 컴포넌트.
"""

import streamlit as st

from common.config import (
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
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
    with st.sidebar:
        st.header("🤖 GraphRAG-Ollama")

        # 1. 문서 처리 섹션
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

        # 2. 통합 모델 설정 섹션
        with st.container(border=True):
            st.subheader("⚙️ 모델 설정")

            # --- 모델 목록 필터링 및 분류 ---
            raw_models = [m for m in (available_models or []) if "---" not in m]

            # 임베딩 관련 키워드 정의
            embed_keywords = ["embed", "bge", "nomic", "mxbai", "snowflake"]

            # [지능형 분류]
            # 1. 임베딩 모델 목록: 키워드 매칭 + 기본 임베딩 모델
            embedding_candidates = [
                m for m in raw_models if any(kw in m.lower() for kw in embed_keywords)
            ]
            actual_embeddings = sorted(
                set(AVAILABLE_EMBEDDING_MODELS + embedding_candidates)
            )
            if DEFAULT_EMBEDDING_MODEL not in actual_embeddings:
                actual_embeddings.append(DEFAULT_EMBEDDING_MODEL)
            actual_embeddings.sort()

            # 2. LLM 모델 목록: 임베딩이 아닌 것 + 기본 LLM 모델
            llm_candidates = [m for m in raw_models if m not in embedding_candidates]
            actual_llms = llm_candidates if llm_candidates else [DEFAULT_OLLAMA_MODEL]
            if DEFAULT_OLLAMA_MODEL not in actual_llms:
                actual_llms.append(DEFAULT_OLLAMA_MODEL)
            actual_llms.sort()

            # --- A. 답변 생성 모델 (LLM) ---
            st.write("**💬 답변 생성 모델 (LLM)**")

            last_model = (
                SessionManager.get("last_selected_model") or DEFAULT_OLLAMA_MODEL
            )
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

            # --- B. 지식 분석 모델 (Embedding) ---
            st.write("**🔍 지식 분석 모델 (Embedding)**")

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

        # 3. 추가 도구/상태 섹션 (필요시)
        with st.expander("🛠️ 고급 설정 및 도구", expanded=False):
            if st.button("🗑️ VRAM 캐시 비우기", use_container_width=True):
                from core.model_loader import ModelManager

                ModelManager.clear_vram()
                st.toast("VRAM이 성공적으로 정리되었습니다.")

            st.caption("리랭커: qwen3 (지능형 채점)")
            st.caption(f"컨텍스트: {OLLAMA_NUM_CTX} tokens")

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button(
                "🗑️ 캐시 및 세션 초기화", use_container_width=True, type="secondary"
            ):
                SessionManager.reset_all_state()
                st.rerun()
