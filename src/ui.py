"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
Clean & Minimal Version: 부가 요소 제거, 직관적인 로딩 및 스트리밍.
"""

import time
import logging
from typing import Callable, Optional

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import fitz  # PyMuPDF

from session import SessionManager
from model_loader import get_available_models
from utils import sync_run
from config import (
    AVAILABLE_EMBEDDING_MODELS,
    OLLAMA_MODEL_NAME,
    UI_CONTAINER_HEIGHT,
    MSG_SIDEBAR_TITLE,
    MSG_PDF_UPLOADER_LABEL,
    MSG_MODEL_SELECTOR_LABEL,
    MSG_EMBEDDING_SELECTOR_LABEL,
    MSG_SYSTEM_STATUS_TITLE,
    MSG_PDF_VIEWER_TITLE,
    MSG_PDF_VIEWER_NO_FILE,
    MSG_PDF_VIEWER_PREV_BUTTON,
    MSG_PDF_VIEWER_NEXT_BUTTON,
    MSG_PDF_VIEWER_ERROR,
    MSG_CHAT_TITLE,
    MSG_CHAT_INPUT_PLACEHOLDER,
    MSG_CHAT_NO_QA_SYSTEM,
    MSG_CHAT_WELCOME,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)

logger = logging.getLogger(__name__)


async def _stream_chat_response(qa_chain, user_input: str, chat_container) -> str:
    """
    미니멀 답변 생성 함수.
    로딩 메시지 표시 후 토큰이 들어오면 답변으로 즉시 대체합니다.
    """
    full_response = ""
    start_time = time.time()
    last_update_time = 0
    update_interval = 0.05
    has_started = False  # 첫 토큰 수신 여부

    current_llm = SessionManager.get("llm")
    if not current_llm:
        return "❌ 오류: LLM 모델이 로드되지 않았습니다."

    run_config = {"configurable": {"llm": current_llm}}
    SessionManager.set("is_generating_answer", True)

    try:
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                # 단 하나의 컨테이너로 로딩과 답변 모두 처리
                answer_container = st.empty()
                answer_container.markdown("⌛ 답변을 생성하고 있습니다...")

                async for event in qa_chain.astream_events(
                    {"input": user_input},
                    config=run_config,
                    version="v1"
                ):
                    kind = event["event"]
                    chunk_text = None

                    # 1. 실시간 토큰 수신 (파서 스트림 우선)
                    if kind == "on_parser_stream":
                        chunk = event["data"].get("chunk")
                        if isinstance(chunk, str):
                            chunk_text = chunk
                    
                    # 2. 백업: 모델 스트림 (파서가 없거나 놓친 경우)
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if hasattr(chunk, "content") and chunk.content:
                            # 이미 파서에서 처리된 경우 중복 방지가 필요할 수 있으나,
                            # StrOutputParser를 쓰면 보통 둘 중 하나만 타거나 내용이 동일함.
                            # 여기서는 파서 이벤트가 없을 때를 대비해 사용
                            if not full_response: # 아주 초기에만 유용할 수 있음 (또는 파서 미사용 시)
                                pass 
                            # 주의: LangChain에서 파서와 모델 스트림이 동시에 발생할 수 있음.
                            # StrOutputParser가 있다면 on_parser_stream만 믿는 것이 중복 출력 방지에 안전함.
                            # 하지만 파서가 동작하지 않는 경우를 위해 남겨둠 (단, 중복 주의)
                            pass

                    if chunk_text:
                        full_response += chunk_text
                        if not has_started:
                            has_started = True
                        
                        now = time.time()
                        if now - last_update_time > update_interval:
                            answer_container.markdown(full_response + "▌")
                            last_update_time = now

                    # 3. 안전 장치: 최종 결과 가로채기 (generate_response 노드 완료 시)
                    if kind == "on_chain_end" and event.get("name") == "generate_response":
                        output = event.get("data", {}).get("output")
                        # GraphState 딕셔너리에서 response 추출
                        if isinstance(output, dict) and "response" in output:
                            final_node_res = output["response"]
                            # 스트리밍된 것보다 최종 결과가 길다면 교체 (누락 방지)
                            if len(final_node_res) > len(full_response):
                                full_response = final_node_res

                # 최종 렌더링 (커서 제거 및 최종 텍스트 확정)
                elapsed_time = time.time() - start_time
                if full_response:
                    answer_container.markdown(full_response)
                    logger.info(f"[UI] 답변 완료: {elapsed_time:.2f}초")
                else:
                    # 상세 로깅 추가
                    llm_model = getattr(current_llm, "model", "Unknown")
                    has_qa_chain = qa_chain is not None
                    
                    log_msg = (
                        f"[UI] 답변 생성 실패 (빈 응답). "
                        f"소요 시간: {elapsed_time:.2f}초, "
                        f"LLM 모델: {llm_model}, "
                        f"QA체인 존재: {has_qa_chain}, "
                        f"이벤트 시작 여부: {has_started}"
                    )
                    logger.warning(log_msg)
                    
                    error_detail = "⚠️ 답변을 생성할 수 없습니다. (응답 값이 비어있습니다.)"
                    if not has_started:
                        error_detail = "⚠️ 답변 생성이 시작되지 않았습니다. Ollama 서버 상태나 모델 설정을 확인해주세요."
                    
                    answer_container.error(error_detail)

        return full_response
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        return f"❌ 오류 발생: {str(e)}"
    finally:
        SessionManager.set("is_generating_answer", False)


def render_sidebar(
    file_uploader_callback: Callable,
    model_selector_callback: Callable,
    embedding_selector_callback: Callable
):
    """
    최소한의 설정만 남긴 사이드바를 렌더링합니다.
    공간 효율을 위해 구분선을 제거하고 Expander를 활용합니다.

    Args:
        file_uploader_callback: 파일 업로드 시 실행될 콜백 함수
        model_selector_callback: LLM 모델 변경 시 실행될 콜백 함수
        embedding_selector_callback: 임베딩 모델 변경 시 실행될 콜백 함수
    """
    with st.sidebar:
        st.header(MSG_SIDEBAR_TITLE)
        
        # 1. 파일 업로드 섹션 (가장 중요하므로 상시 노출)
        st.file_uploader(
            MSG_PDF_UPLOADER_LABEL, 
            type="pdf", 
            key="pdf_uploader", 
            on_change=file_uploader_callback
        )

        # 2. 모델 설정 섹션 (접이식으로 공간 절약)
        # expanded=False로 설정하여 기본적으로는 숨김 처리 (사용자 경험에 따라 변경 가능)
        with st.expander("⚙️ 모델 설정", expanded=False):
            # 모델 목록 가져오기
            available_models = get_available_models()
            is_ollama_error = (
                bool(available_models) and 
                available_models[0] == MSG_ERROR_OLLAMA_NOT_RUNNING
            )
            actual_models = [] if is_ollama_error else [m for m in available_models if "---" not in m]
            
            last_model = SessionManager.get("last_selected_model")
            if not last_model or (actual_models and last_model not in actual_models):
                last_model = actual_models[0] if actual_models else OLLAMA_MODEL_NAME
                SessionManager.set("last_selected_model", last_model)

            try:
                idx = available_models.index(last_model)
            except ValueError:
                idx = 0

            # 긴 모델 이름이 잘리는 문제를 위해 help 툴팁 추가
            st.selectbox(
                MSG_MODEL_SELECTOR_LABEL, 
                available_models, 
                index=idx, 
                key="model_selector", 
                on_change=model_selector_callback, 
                disabled=is_ollama_error,
                help="사용할 LLM 모델을 선택하세요."
            )

            # 임베딩 모델 선택
            last_emb = SessionManager.get("last_selected_embedding_model") or AVAILABLE_EMBEDDING_MODELS[0]
            try:
                emb_idx = AVAILABLE_EMBEDDING_MODELS.index(last_emb)
            except ValueError:
                emb_idx = 0
                
            st.selectbox(
                MSG_EMBEDDING_SELECTOR_LABEL, 
                AVAILABLE_EMBEDDING_MODELS, 
                index=emb_idx, 
                key="embedding_model_selector", 
                on_change=embedding_selector_callback,
                help="문서 검색에 사용할 임베딩 모델입니다."
            )
        
        # 3. 시스템 상태 섹션 (구분선 없이 여백으로 분리)
        st.markdown("#### " + MSG_SYSTEM_STATUS_TITLE)
        
        # 상태 메시지를 표시할 빈 컨테이너 반환
        return st.container()


def render_pdf_viewer():
    _pdf_viewer_fragment()


@st.fragment
def _pdf_viewer_fragment():
    """PDF 뷰어 (Fragment) - 개선된 네비게이션"""
    st.subheader(MSG_PDF_VIEWER_TITLE)
    
    pdf_bytes = SessionManager.get("pdf_file_bytes")
    if not pdf_bytes:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return
    
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
            
            # 세션 상태 초기화
            if "current_page" not in st.session_state: 
                st.session_state.current_page = 1
            
            # 페이지 범위 보정 (파일 변경 등으로 총 페이지가 줄었을 때)
            if st.session_state.current_page > total_pages:
                st.session_state.current_page = 1
            if st.session_state.current_page < 1:
                st.session_state.current_page = 1

            # --- PDF 뷰어 렌더링 ---
            pdf_viewer(
                input=pdf_bytes, 
                height=UI_CONTAINER_HEIGHT, 
                pages_to_render=[st.session_state.current_page]
            )

            # --- 하단 네비게이션 ---
            # 헬퍼 함수: 페이지 변경 콜백
            def go_prev():
                if st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
            
            def go_next():
                if st.session_state.current_page < total_pages:
                    st.session_state.current_page += 1

            # 레이아웃: [이전] [페이지 입력 / 총페이지] [다음]
            c1, c2, c3 = st.columns([1, 1, 1])
            
            with c1:
                # 이전 페이지 버튼 (on_click 사용)
                st.button(
                    MSG_PDF_VIEWER_PREV_BUTTON, 
                    key="btn_pdf_prev", 
                    use_container_width=True,
                    disabled=(st.session_state.current_page <= 1),
                    on_click=go_prev
                )

            with c2:
                # 가운데 병합된 '현재 / 총 page' 레이아웃
                p1, p2 = st.columns([1, 1])
                with p1:
                    def update_page_input():
                        # number_input의 값이 변경되면 session_state에 반영
                        # key가 'num_input_page'이므로 st.session_state.num_input_page에 값이 있음
                        st.session_state.current_page = int(st.session_state.num_input_page)

                    st.number_input(
                        "페이지 이동", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=st.session_state.current_page, 
                        label_visibility="collapsed",
                        key="num_input_page",
                        on_change=update_page_input
                    )
                with p2:
                    st.markdown(
                        f"<div style='line-height: 2.3em; font-size: 1.0em;'>"
                        f"&nbsp;/ {total_pages} pages</div>", 
                        unsafe_allow_html=True
                    )

            with c3:
                # 다음 페이지 버튼 (on_click 사용)
                st.button(
                    MSG_PDF_VIEWER_NEXT_BUTTON, 
                    key="btn_pdf_next", 
                    use_container_width=True,
                    disabled=(st.session_state.current_page >= total_pages),
                    on_click=go_next
                )
            
    except Exception as e:
        logger.error(f"PDF 뷰어 오류: {e}", exc_info=True)
        st.error(f"PDF 오류: {e}")


def render_left_column():
    _chat_fragment()


def render_message(role: str, content: str):
    """
    메시지를 역할에 따라 스타일링하여 렌더링합니다.
    Args:
        role: 메시지 작성자 역할 ('user' 또는 'assistant')
        content: 메시지 본문
    """
    # 아바타 설정: 사용자(👤), 어시스턴트(🤖)
    avatar_icon = "🤖" if role == "assistant" else "👤"
    
    with st.chat_message(role, avatar=avatar_icon):
        # 답변 내의 출처(Sources) 부분을 시각적으로 분리
        # 가정: LLM 답변에 '출처:' 또는 'Sources:' 라는 구분자가 있다고 가정
        # 실제 프롬프트에 따라 구분자는 달라질 수 있음
        
        # 간단한 파싱 로직 (필요시 정규표현식으로 고도화 가능)
        separator = None
        if "출처:" in content:
            separator = "출처:"
        elif "Sources:" in content:
            separator = "Sources:"
            
        if role == "assistant" and separator:
            try:
                parts = content.split(separator, 1)
                main_content = parts[0].strip()
                sources = parts[1].strip()
                
                st.markdown(main_content)
                if sources:
                    with st.expander("📚 참고 문헌 (Sources)", expanded=False):
                        st.markdown(f"**{separator}**\n{sources}")
            except Exception:
                # 파싱 에러 시 원본 그대로 출력
                st.markdown(content)
        else:
            st.markdown(content)


@st.fragment
def _chat_fragment():
    """채팅 구역 (Fragment)"""
    st.subheader(MSG_CHAT_TITLE)
    chat_container = st.container(height=UI_CONTAINER_HEIGHT, border=True)

    messages = SessionManager.get_messages()
    for msg in messages:
        with chat_container:
            render_message(msg["role"], msg["content"])

    if not messages:
        with chat_container: 
            st.info(MSG_CHAT_WELCOME)

    is_gen = SessionManager.get("is_generating_answer")
    if user_input := st.chat_input(MSG_CHAT_INPUT_PLACEHOLDER, disabled=is_gen, key="chat_input_clean"):
        SessionManager.add_message("user", user_input)
        
        # 즉시 사용자 메시지 렌더링
        with chat_container:
            render_message("user", user_input)

        qa_chain = SessionManager.get("qa_chain")
        if qa_chain:
            final_ans = sync_run(_stream_chat_response(qa_chain, user_input, chat_container))
            if final_ans and not final_ans.startswith("❌"):
                SessionManager.add_message("assistant", final_ans)
                st.rerun()
        else:
            st.error(MSG_CHAT_NO_QA_SYSTEM)
