import torch
# 아래 코드는 특정 PyTorch/Torchvision 버전 간 호환성 문제로 인해 torchvision.ops 등을 찾지 못하는 오류를
# 해결하기 위한 임시 조치입니다. (예: torchvision 로딩 시 `torch.classes.load_library` 관련 오류)
torch.classes.__path__ = []
import tempfile
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import logging
import time
from utils import (
    SessionManager,
    get_ollama_models,
    load_llm,
    process_pdf,
    update_qa_chain as util_update_qa_chain,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)
SessionManager.init_session()

def handle_model_change(selected_model: str):
    """모델 변경 처리"""
    if not selected_model or selected_model == st.session_state.get("last_selected_model"):
        return

    old_model = SessionManager.update_model(selected_model)
    logging.info(f"LLM 변경 감지: {old_model} -> {selected_model}")

    if not st.session_state.get("pdf_processed"):
        logging.info(f"모델 선택 변경됨 (PDF 미처리 상태): {selected_model}")
        return

    try:
        # 1. 새 LLM 로드
        with st.spinner(f"'{selected_model}' 모델 로딩 중..."):
            st.session_state.llm = load_llm(selected_model)
        # 2. QA 체인 업데이트
        if st.session_state.get("vector_store") and st.session_state.get("llm"):
            with st.spinner("QA 시스템 업데이트 중..."):
                st.session_state.qa_chain = util_update_qa_chain(
                    st.session_state.llm,
                    st.session_state.vector_store
                )
                logging.info(f"'{selected_model}' 모델로 QA 체인 업데이트 완료.")
                success_message = f"✅ '{selected_model}' 모델로 변경이 완료되었습니다."
                SessionManager.add_message("assistant", success_message)
                st.session_state.last_model_change_message = success_message
        else:
            raise ValueError("벡터 저장소 또는 LLM을 찾을 수 없습니다. PDF 재처리가 필요할 수 있습니다.")

    except Exception as e:
        error_msg = f"모델 변경 중 오류 발생: {e}"
        logging.error(f"{error_msg} ({selected_model})", exc_info=True)
        SessionManager.reset_session_state(["llm", "qa_chain"])
        SessionManager.add_message("assistant", f"❌ {error_msg}")
        st.session_state.last_model_change_message = f"❌ {error_msg}"

def save_uploaded_file(uploaded_file) -> str | None:
    """UploadFile 객체를 임시 파일로 저장하고 경로를 반환합니다."""
    try:
        with st.spinner(f"'{uploaded_file.name}' 파일 저장 중..."):
            temp_dir = tempfile.gettempdir()
            # 파일 이름에 타임스탬프를 추가하여 고유성 보장
            timestamp = int(time.time())
            safe_filename = f"rag_chatbot_{timestamp}_{uploaded_file.name}"
            temp_pdf_path = os.path.join(temp_dir, safe_filename)
            
            with open(temp_pdf_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            logging.info(f"임시 PDF 파일 생성 성공: {temp_pdf_path}")
            return temp_pdf_path
    except Exception as e:
        logging.error(f"임시 파일 저장 실패: {e}", exc_info=True)
        st.error(f"임시 파일 저장에 실패했습니다: {e}")
        return None

def handle_file_upload_and_process(uploaded_file):
    """PDF 파일 업로드와 처리를 한 번에 관리"""
    if not uploaded_file or uploaded_file.name == st.session_state.get("last_uploaded_file_name"):
        return

    # 1. 이전 임시 파일 정리 (Best-effort)
    if st.session_state.get("temp_pdf_path") and os.path.exists(st.session_state.temp_pdf_path):
        try:
            os.remove(st.session_state.temp_pdf_path)
            logging.info("이전 임시 PDF 파일 삭제 성공")
        except Exception as e:
            logging.warning(f"이전 임시 PDF 파일 삭제 실패: {e}")

    # 2. 세션 상태 리셋 및 처리 시작 플래그 설정
    SessionManager.reset_for_new_file(uploaded_file)
    st.session_state.pdf_is_processing = True

    try:
        # 3. 새 PDF 파일 저장
        temp_pdf_path = save_uploaded_file(uploaded_file)
        if not temp_pdf_path:
            # 파일 저장 실패 시 오류 메시지는 save_uploaded_file에서 이미 표시됨
            SessionManager.set_error_state("임시 파일 저장에 실패하여 처리를 중단합니다.")
            return

        st.session_state.temp_pdf_path = temp_pdf_path
        st.session_state.current_file_path = temp_pdf_path
        st.session_state["pdf_viewer_key"] = f"pdf_viewer_{uploaded_file.name}_{int(time.time())}"
        
        # 4. 문서 처리 시작 메시지 표시
        SessionManager.add_message(
            "assistant", 
            f"📂 '{uploaded_file.name}' 파일 업로드 완료.\n\n"
            f"⏳ 문서 처리를 시작합니다. 잠시만 기다려주세요..."
        )
        
        # 5. PDF 처리 실행
        current_selected_model = st.session_state.get("last_selected_model")
        if not current_selected_model:
            st.warning("모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
            SessionManager.set_error_state("모델이 선택되지 않아 PDF 처리를 진행할 수 없습니다.")
            return

        process_pdf(uploaded_file, current_selected_model, st.session_state.temp_pdf_path)

    except Exception as e:
        error_msg = f"파일 업로드 또는 처리 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        SessionManager.set_error_state(error_msg)
    finally:
        st.session_state.pdf_is_processing = False
        # 모든 과정이 끝난 후 UI 전체를 최종 상태로 업데이트하기 위해 단 한번 rerun
        st.rerun()

# --- Constants ---
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"
MSG_PREPARING_ANSWER = "답변 생성 준비 중..."
MSG_THINKING = "🤔 생각을 정리하는 중입니다..."
MSG_WRITING_ANSWER = "답변을 작성하는 중..."
MSG_NO_THOUGHT_PROCESS = "아직 생각 과정이 없습니다."
MSG_NO_RELATED_INFO = "죄송합니다, 제공된 문서에서 관련 정보를 찾을 수 없었습니다."


def process_chat_response(qa_chain, user_input, chat_container):
    """
    스트리밍 방식으로 LLM 응답을 처리하고 채팅 컨테이너에 표시하는 함수
    """
    with chat_container, st.chat_message("assistant"):
        thought_expander = st.expander("🤔 생각 과정", expanded=False)
        thought_placeholder = thought_expander.empty()
        message_placeholder = st.empty()
        
        message_placeholder.markdown(MSG_PREPARING_ANSWER)
        thought_placeholder.markdown(MSG_NO_THOUGHT_PROCESS)

        try:
            start_time = time.time()
            thought_buffer = ""
            response_buffer = ""
            is_thinking = False
            
            for chunk in qa_chain.stream({"input": user_input}):
                answer_chunk = chunk.get("answer", "")
                if not answer_chunk:
                    continue
                
                # <think> 태그 처리
                if THINK_START_TAG in answer_chunk:
                    is_thinking = True
                    # <think> 태그 이후의 내용을 생각 버퍼에 추가
                    # 이전 청크의 </think> 와 같은 청크에 <think> 가 오는 경우를 대비해 split 사용
                    parts = answer_chunk.split(THINK_START_TAG, 1)
                    response_buffer += parts[0] # <think> 이전 내용은 답변으로 간주
                    thought_buffer = parts[1]
                    message_placeholder.markdown(MSG_THINKING)
                    continue
                
                # </think> 태그 처리
                if THINK_END_TAG in answer_chunk:
                    is_thinking = False
                    parts = answer_chunk.split(THINK_END_TAG, 1)
                    thought_buffer += parts[0] # </think> 이전 내용은 생각으로 간주
                    
                    if thought_buffer.strip():
                        thought_placeholder.markdown(thought_buffer)
                    
                    response_buffer += parts[1] # </think> 이후 내용은 답변으로 간주
                    message_placeholder.markdown(MSG_WRITING_ANSWER)
                    continue
                
                # 스트리밍 내용 표시
                if is_thinking:
                    thought_buffer += answer_chunk
                    thought_placeholder.markdown(thought_buffer + "▌")
                else:
                    response_buffer += answer_chunk
                    message_placeholder.markdown(response_buffer + "▌")
            
            # 스트리밍 종료 후 최종 내용 표시
            if thought_buffer.strip():
                thought_placeholder.markdown(thought_buffer)
            
            final_answer = response_buffer.strip()
            if not final_answer:
                final_answer = MSG_NO_RELATED_INFO
            
            message_placeholder.markdown(final_answer)
            SessionManager.add_message("assistant", final_answer)

            end_time = time.time()
            logging.info(f"LLM 답변 생성 완료 (소요 시간: {end_time - start_time:.2f}초)")

        except Exception as e:
            error_msg = f"답변 생성 중 오류 발생: {str(e)}"
            logging.error(error_msg, exc_info=True)
            message_placeholder.error(error_msg)
            SessionManager.add_message("assistant", f"❌ {error_msg}")

def render_sidebar():
    """사이드바 UI를 렌더링하고 사용자 입력을 처리합니다."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # 모델 선택
        selected_model = None
        try:
            models = get_ollama_models()
            last_model = st.session_state.get("last_selected_model")
            current_model_index = models.index(last_model) if last_model and last_model in models else 0
            if models:
                selected_model = st.selectbox(
                    "Select an Ollama model",
                    models,
                    index=current_model_index,
                    key="model_selector"
                )
            else:
                st.text("Failed to load Ollama models.")
        except Exception as e:
            st.error(f"Failed to load Ollama models: {e}")
            st.warning("Ollama가 설치되어 있는지, Ollama 서버가 실행 중인지 확인해주세요.")

        if selected_model and selected_model != st.session_state.get("last_selected_model"):
            handle_model_change(selected_model)

        # 파일 업로더
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        handle_file_upload_and_process(uploaded_file)
        
        st.divider()
        
        # PDF 뷰어 설정
        st.session_state.resolution_boost = st.slider("Resolution boost", 1, 10, 1)
        st.session_state.pdf_width = st.slider("PDF width", 100, 1000, 1000)
        st.session_state.pdf_height = st.slider("PDF height", 100, 10000, 1000)

def render_pdf_viewer_column():
    """PDF 뷰어 컬럼을 렌더링합니다."""
    st.subheader("📄 PDF Preview")
    
    if st.session_state.get("temp_pdf_path") and os.path.exists(st.session_state.temp_pdf_path):
        try:
            viewer_key = st.session_state.get("pdf_viewer_key", "pdf_viewer_default")
            pdf_viewer(
                input=st.session_state.temp_pdf_path,
                width=st.session_state.pdf_width,
                height=st.session_state.pdf_height,
                key=viewer_key,
                resolution_boost=st.session_state.resolution_boost
            )
        except Exception as e:
            error_msg = f"PDF 미리보기 중 오류 발생: {str(e)}"
            logging.error(error_msg, exc_info=True)
            st.error(error_msg)
            # PDF 뷰어 복구 시도
            if st.button("PDF 뷰어 재시도"):
                st.session_state["pdf_viewer_key"] = f"pdf_viewer_retry_{int(time.time())}"
                st.rerun()

def render_chat_column():
    """채팅 컬럼을 렌더링하고 채팅 로직을 처리합니다."""
    st.subheader("💬 Chat")
    
    chat_container = st.container(height=650, border=True)
    
    with chat_container:
        # 기존 메시지 표시
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)
    
    # 사용자 입력 처리
    if user_input := st.chat_input(
        "PDF 내용에 대해 질문해보세요.",
        key='user_input',
        disabled=not SessionManager.is_ready_for_chat()
    ):
        SessionManager.add_message("user", user_input)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # QA 체인으로 응답 생성
        qa_chain = st.session_state.get("qa_chain")
        if not qa_chain:
            error_message = "❌ QA 시스템이 준비되지 않았습니다. 모델 변경이 진행 중이거나 PDF 처리가 필요할 수 있습니다."
            SessionManager.add_message("assistant", error_message)
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(error_message)
        else:
            try:
                process_chat_response(qa_chain, user_input, chat_container)
            except Exception as e:
                error_message = f"❌ 응답 생성 중 오류가 발생했습니다: {str(e)}"
                SessionManager.add_message("assistant", error_message)
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                logging.error(f"응답 생성 오류: {e}", exc_info=True)

def main():
    """메인 애플리케이션 실행 함수"""
    render_sidebar()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_chat_column()

    with col_right:
        render_pdf_viewer_column()

if __name__ == "__main__":
    main()