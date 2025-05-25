import torch
torch.classes.__path__ = [] # PyTorch/torchvision 특정 버전 호환성 문제 해결을 위한 임시 조치일 수 있음
import tempfile
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import logging
import json
import re
import html
import time # 시간 측정을 위해 time 모듈 import
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
    # st.rerun() # selectbox 값 변경으로 인한 자동 rerun으로 충분하므로 명시적 rerun 제거

def handle_pdf_upload(uploaded_file):
    """PDF 파일 업로드 처리"""
    if not uploaded_file:
        return

    # 같은 파일이 다시 업로드된 경우 처리하지 않음
    if uploaded_file.name == st.session_state.get("last_uploaded_file_name"):
        return

    try:
        # 1. 이전 PDF 파일 정리
        if st.session_state.get("temp_pdf_path") and os.path.exists(st.session_state.temp_pdf_path):
            try:
                os.remove(st.session_state.temp_pdf_path)
                logging.info("이전 임시 PDF 파일 삭제 성공")
            except Exception as e:
                logging.warning(f"이전 임시 PDF 파일 삭제 실패: {e}")

        # 2. 세션 상태 리셋 (파일 저장 전에 실행)
        SessionManager.reset_for_new_file(uploaded_file)
        
        # 3. 새 PDF 파일을 임시 디렉토리에 저장
        temp_dir = tempfile.gettempdir()
        temp_pdf_path = os.path.join(temp_dir, f"rag_chatbot_{int(time.time())}_{uploaded_file.name}")
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # 4. 세션 상태 업데이트
        st.session_state.temp_pdf_path = temp_pdf_path
        st.session_state.current_file_path = temp_pdf_path  # 현재 파일 경로 설정
        logging.info(f"임시 PDF 파일 생성 성공: {temp_pdf_path}")
        
        SessionManager.add_message(
            "assistant", (
                f"📂 새 PDF 파일 '{uploaded_file.name}'이(가) 업로드되었습니다.\n\n"
                "잠시만 기다려주세요."
                )
        )
        
        # PDF 뷰어 키 업데이트
        st.session_state["pdf_viewer_key"] = f"pdf_viewer_{uploaded_file.name}_{int(time.time())}"
        
        # 새 파일 정보로 UI를 업데이트하기 위해 rerun
        st.rerun()
        
    except Exception as e:
        error_msg = f"임시 PDF 파일 생성 실패: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        st.session_state.temp_pdf_path = None

def handle_pdf_processing(uploaded_file):
    """PDF 처리 상태 관리 및 실행"""
    if not (uploaded_file and st.session_state.temp_pdf_path):
        return

    if (st.session_state.get("pdf_processed") or 
        st.session_state.get("pdf_processing_error") or 
        st.session_state.get("pdf_is_processing")):
        return

    current_selected_model = st.session_state.get("last_selected_model")
    if not current_selected_model:
        st.warning("모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
        return

    st.session_state.pdf_is_processing = True
    SessionManager.add_message("assistant", f"⏳ '{uploaded_file.name}' 문서 처리 중...")
    
    try:
        process_pdf(uploaded_file, current_selected_model, st.session_state.temp_pdf_path)
    except Exception as e:
        error_msg = f"PDF 처리 중 오류 발생: {e}"
        logging.error(error_msg)
        SessionManager.set_error_state(error_msg)
    finally:
        st.session_state.pdf_is_processing = False
        # PDF 처리 시도 후 (성공/실패 모두) UI 업데이트를 위해 rerun
        st.rerun()

def _parse_llm_output(full_llm_output: str) -> tuple[str, str, dict | None]:
    """LLM의 전체 출력에서 생각 과정을 분리하고, 나머지 텍스트를 직접 반환합니다."""
    thought_content = ""
    remaining_output = full_llm_output.strip()

    if remaining_output.startswith("<think>"):
        think_end_tag = "</think>"
        think_end_idx = remaining_output.find(think_end_tag)
        if think_end_idx != -1:
            thought_content = remaining_output[len("<think>"):think_end_idx].strip()
            remaining_output = remaining_output[think_end_idx + len(think_end_tag):].strip()
    
    return thought_content, remaining_output, None

def process_chat_response(qa_chain, user_input, chat_container):
    """채팅 응답 처리"""
    with chat_container:
        with st.chat_message("assistant"):
            thought_expander = st.expander("🤔 생각 과정", expanded=False)
            thought_placeholder = thought_expander.empty()  # 생각 과정을 표시할 영역
            message_placeholder = st.empty()  # 답변을 표시할 영역
            message_placeholder.write("답변 생성 시작...")  # 초기 메시지

            try:
                logging.info("답변 생성 시작...")
                start_time = time.time()

                # LLM 응답을 스트리밍하여 실시간으로 표시
                full_llm_output = ""
                thought_content = ""
                is_thinking = False
                current_content = ""
                
                for chunk_text in qa_chain.stream({"input": user_input}):
                    full_llm_output += chunk_text
                    
                    # <think> 태그가 시작되는지 확인
                    if "<think>" in chunk_text and not is_thinking:
                        is_thinking = True
                        current_content = ""
                        message_placeholder.write("생각 중...") # 생각 과정 시작 시 상태 메시지 변경
                        continue
                    
                    # </think> 태그가 있는지 확인
                    if "</think>" in chunk_text and is_thinking:
                        is_thinking = False
                        thought_content = current_content
                        thought_placeholder.markdown(thought_content + "▌")
                        current_content = ""
                        message_placeholder.write("답변 생성 중...") # 답변 생성 시작 시 상태 메시지 변경
                        continue
                    
                    # 현재 상태에 따라 적절한 placeholder에 내용 추가
                    if is_thinking:
                        current_content += chunk_text
                        thought_placeholder.markdown(current_content + "▌")
                    else:
                        current_content += chunk_text
                        message_placeholder.markdown(current_content + "▌")

                # 최종 내용 표시
                if thought_content:
                    thought_placeholder.markdown(thought_content)
                message_placeholder.markdown(current_content)

                end_time = time.time()
                generation_time = end_time - start_time
                logging.info(f"LLM 답변 생성 완료 (소요 시간: {generation_time:.2f}초)")

                if not full_llm_output:
                    raise ValueError("LLM으로부터 빈 응답을 받았습니다.")
                # 2. LLM 출력 파싱 (생각 과정, raw JSON, 파싱된 데이터)
                # parsed_json_data will now always be None
                parsed_thought_final, direct_llm_text_output, parsed_json_data = _parse_llm_output(full_llm_output)

                # 3. 최종 생각 과정 표시 (스트리밍 중에는 표시하지 않음)
                if parsed_thought_final:
                    thought_expander.markdown(parsed_thought_final)
                else:
                    thought_expander.empty()

                # 4. 답변 처리 (이제 direct_llm_text_output을 직접 사용)
                # parsed_json_data는 항상 None이므로, 이전의 JSON 파싱 성공/실패 분기 로직은 필요 없음.
                if not direct_llm_text_output:
                    message_placeholder.markdown("LLM으로부터 답변 내용을 받지 못했습니다.")
                    SessionManager.add_message("assistant", "LLM으로부터 답변 내용을 받지 못했습니다.")
                    return
                
                message_placeholder.markdown(direct_llm_text_output, unsafe_allow_html=True)
                SessionManager.add_message("assistant", direct_llm_text_output)

                # "참고 자료" 출력 로직 제거
                # llm_provided_sources 변수는 여전히 존재하지만 UI에 표시하지 않음

            except Exception as e:
                logging.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
                error_message = f"❌ 답변 생성 중 오류가 발생했습니다: {e}"
                message_placeholder.error(error_message)
                SessionManager.add_message("assistant", error_message)

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
        st.divider()
        resolution_boost = st.slider("Resolution boost", 1, 10, 1)
        width = st.slider("PDF width", 100, 1000, 1000)
        height = st.slider("PDF height", 100, 10000, 1000)

    col_left, col_right = st.columns([1, 1])

    with col_right:
        st.subheader("📄 PDF Preview")
        handle_pdf_upload(uploaded_file)
        
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
            else:
                st.warning("PDF 파일을 로드할 수 없습니다. 다시 업로드해주세요.")

    with col_left:
        st.subheader("💬 Chat")
        chat_container = st.container(height=500, border=True)
        display_chat_messages(chat_container)

        if not st.session_state.get("pdf_processed"):
            handle_pdf_processing(uploaded_file)
            
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