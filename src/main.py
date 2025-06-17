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
    RETRIEVER_CONFIG,
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

def handle_file_upload_and_process(uploaded_file):
    """PDF 파일 업로드와 처리를 한 번에 관리"""
    if not uploaded_file:
        return

    # 같은 파일이 다시 업로드된 경우 처리하지 않음
    if uploaded_file.name == st.session_state.get("last_uploaded_file_name"):
        return

    # 1. 이전 임시 파일 정리 (Best-effort)
    if st.session_state.get("temp_pdf_path") and os.path.exists(st.session_state.temp_pdf_path):
        try:
            os.remove(st.session_state.temp_pdf_path)
            logging.info("이전 임시 PDF 파일 삭제 성공")
        except Exception as e:
            logging.warning(f"이전 임시 PDF 파일 삭제 실패: {e}")

    # 2. 세션 상태 리셋
    SessionManager.reset_for_new_file(uploaded_file)
    st.session_state.pdf_is_processing = True

    try:
        # 3. 새 PDF 파일을 임시 디렉토리에 저장
        with st.spinner(f"'{uploaded_file.name}' 파일 저장 중..."):
            temp_dir = tempfile.gettempdir()
            temp_pdf_path = os.path.join(temp_dir, f"rag_chatbot_{int(time.time())}_{uploaded_file.name}")
            with open(temp_pdf_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            st.session_state.temp_pdf_path = temp_pdf_path
            st.session_state.current_file_path = temp_pdf_path
            st.session_state["pdf_viewer_key"] = f"pdf_viewer_{uploaded_file.name}_{int(time.time())}"
            logging.info(f"임시 PDF 파일 생성 성공: {temp_pdf_path}")
        
        # 4. 문서 처리 시작 메시지 표시
        SessionManager.add_message(
            "assistant", 
            f"📂 '{uploaded_file.name}' 파일 업로드 완료.\n\n"
            f"⏳ 문서 처리를 시작합니다. 잠시만 기다려주세요..."
        )
        
        # 5. PDF 처리 실행 (rerun 없이 바로 실행)
        current_selected_model = st.session_state.get("last_selected_model")
        if not current_selected_model:
            st.warning("모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
            # 모델 미선택 시 처리를 중단하고 사용자에게 알림
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

def process_chat_response(qa_chain, user_input, chat_container):
    """채팅 응답 처리"""
    with chat_container:
        with st.chat_message("assistant"):
            thought_expander = st.expander("🤔 생각 과정", expanded=False)
            thought_placeholder = thought_expander.empty()
            message_placeholder = st.empty()
            message_placeholder.markdown("답변 생성 준비 중...")

            try:
                start_time = time.time()
                thought_buffer = ""
                response_buffer = ""
                is_thinking = False
                update_counter = 0
                
                for chunk in qa_chain.stream({"input": user_input}):
                    if isinstance(chunk, dict):
                        chunk_text = chunk.get("answer", "")
                    else:
                        chunk_text = str(chunk)
                        
                    if not chunk_text:
                        continue
                        
                    # 태그 처리
                    if "<think>" in chunk_text:
                        is_thinking = True
                        thought_buffer = chunk_text.split("<think>")[1]
                        # 생각 과정 시작 시 메시지 업데이트
                        message_placeholder.markdown("🤔 생각을 정리하는 중입니다...")
                        continue
                        
                    if "</think>" in chunk_text:
                        is_thinking = False
                        thought_end_idx = chunk_text.find("</think>")
                        thought_buffer += chunk_text[:thought_end_idx]
                        response_buffer = chunk_text[thought_end_idx + len("</think>"):]
                        if thought_buffer.strip():
                            thought_placeholder.markdown(thought_buffer)
                        # 생각 과정 종료 시 메시지 업데이트
                        message_placeholder.markdown("답변을 작성하는 중...")
                        continue
                    
                    # 버퍼에 추가 및 화면 업데이트
                    if is_thinking:
                        thought_buffer += chunk_text
                        if update_counter % 3 == 0:
                            thought_placeholder.markdown(thought_buffer + "▌")
                    else:
                        response_buffer += chunk_text
                        if update_counter % 3 == 0:
                            message_placeholder.markdown(response_buffer + "▌")
                    
                    update_counter += 1
                    # 스트리밍 효과를 시각적으로 부드럽게 하기 위한 짧은 대기입니다.
                    # 응답 속도가 매우 중요한 경우 이 값을 줄이거나 제거할 수 있습니다.
                    time.sleep(0.01)  
                
                # 최종 응답 표시
                if thought_buffer.strip():
                    thought_placeholder.markdown(thought_buffer)
                if response_buffer.strip():
                    message_placeholder.markdown(response_buffer)
                
                # 세션에 저장
                SessionManager.add_message("assistant", response_buffer)
                
                end_time = time.time()
                logging.info(f"LLM 답변 생성 완료 (소요 시간: {end_time - start_time:.2f}초)")

            except Exception as e:
                error_msg = f"답변 생성 중 오류 발생: {str(e)}"
                logging.error(error_msg, exc_info=True)
                message_placeholder.error(error_msg)
                SessionManager.add_message("assistant", f"❌ {error_msg}")

def display_chat_messages(chat_container):
    """채팅 컨테이너에 모든 메시지를 표시"""
    with chat_container:
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

def main():
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ Settings")
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

        if (
            selected_model
            and selected_model != st.session_state.get("last_selected_model")
        ):
            handle_model_change(selected_model)

        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        # 파일 업로드가 감지되면 바로 처리 함수 호출
        handle_file_upload_and_process(uploaded_file)
        
        st.divider()
        resolution_boost = st.slider("Resolution boost", 1, 10, 1)
        width = st.slider("PDF width", 100, 1000, 1000)
        height = st.slider("PDF height", 100, 10000, 1000)

    col_left, col_right = st.columns([1, 1])

    # 메인 컨테이너 설정
    with col_right:
        st.subheader("📄 PDF Preview")
        
        # PDF 뷰어 렌더링
        if uploaded_file:
            if st.session_state.get("temp_pdf_path") and os.path.exists(st.session_state.temp_pdf_path):
                try:
                    viewer_key = st.session_state.get("pdf_viewer_key", "pdf_viewer_default")
                    pdf_viewer(
                        input=st.session_state.temp_pdf_path,
                        width=width,
                        height=height,
                        key=viewer_key,
                        resolution_boost=resolution_boost
                    )
                    logging.info(f"PDF 뷰어 렌더링 성공 - 키: {viewer_key}")
                except Exception as e:
                    error_msg = f"PDF 미리보기 중 오류 발생: {str(e)}"
                    logging.error(error_msg, exc_info=True)
                    st.error(error_msg)
                    
                    # PDF 뷰어 복구 시도
                    try:
                        st.session_state["pdf_viewer_key"] = f"pdf_viewer_retry_{int(time.time())}"
                        st.rerun()
                    except Exception as retry_error:
                        logging.error(f"PDF 뷰어 복구 실패: {retry_error}")

    # 채팅 컨테이너 설정
    with col_left:
        st.subheader("💬 Chat")
        
        chat_container = st.container(height=650, border=True)
        display_chat_messages(chat_container)
            
        user_input = st.chat_input(
            "PDF 내용에 대해 질문해보세요.",
            key='user_input',
            disabled=not SessionManager.is_ready_for_chat()
        )

        # 새 메시지 처리
        if user_input and SessionManager.is_ready_for_chat(): # is_ready_for_chat 추가
            SessionManager.add_message("user", user_input)
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
            qa_chain = st.session_state.get("qa_chain")
            if not qa_chain:
                error_message = "❌ QA 시스템이 준비되지 않았습니다. 모델 변경이 진행 중이거나 PDF 처리가 필요할 수 있습니다."
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                SessionManager.add_message("assistant", error_message)
            else:
                try:
                    process_chat_response(qa_chain, user_input, chat_container)
                except Exception as e:
                    error_message = f"❌ 응답 생성 중 오류가 발생했습니다: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    SessionManager.add_message("assistant", error_message)
                    logging.error(f"응답 생성 오류: {e}", exc_info=True)

if __name__ == "__main__":
    main()