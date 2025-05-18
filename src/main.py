import torch
torch.classes.__path__ = [] # 호환성 문제 해결을 위한 임시 조치
import tempfile
import os
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from streamlit_pdf_viewer import pdf_viewer
import logging
from utils import (
    SessionManager,
    get_ollama_models,
    load_llm,
    QA_PROMPT,
    process_pdf,
    RETRIEVER_CONFIG,  # 리트리버 설정 상수 import
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 페이지 설정
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

# 세션 상태 초기화
SessionManager.init_session()

def update_qa_chain(llm, vector_store):
    """QA 체인 업데이트"""
    try:
        combine_chain = create_stuff_documents_chain(llm, QA_PROMPT)
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG['search_type'],
            search_kwargs=RETRIEVER_CONFIG['search_kwargs']
        )
        return create_retrieval_chain(retriever, combine_chain)
    except Exception as e:
        raise ValueError(f"QA 체인 업데이트 실패: {e}")

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
                st.session_state.qa_chain = update_qa_chain(
                    st.session_state.llm,
                    st.session_state.vector_store
                )
                logging.info(f"'{selected_model}' 모델로 QA 체인 업데이트 완료.")
        else:
            raise ValueError("벡터 저장소 또는 LLM을 찾을 수 없습니다. PDF 재처리가 필요할 수 있습니다.")

    except Exception as e:
        error_msg = f"모델 변경 중 오류 발생: {e}"
        logging.error(f"{error_msg} ({selected_model})", exc_info=True)
        SessionManager.reset_session_state(["llm", "qa_chain", "pdf_processed"])
        SessionManager.add_message("assistant", f"❌ {error_msg}")
        
    st.rerun()  # 직접 rerun 호출

def handle_pdf_upload(uploaded_file):
    """PDF 파일 업로드 처리"""
    if not uploaded_file:
        return

    if uploaded_file.name == st.session_state.get("last_uploaded_file_name"):
        return

    try:
        # 1. 이전 PDF 파일 정리
        if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
            try:
                os.remove(st.session_state.temp_pdf_path)
                logging.info("이전 임시 PDF 파일 삭제 성공")
            except Exception as e:
                logging.warning(f"이전 임시 PDF 파일 삭제 실패: {e}")

        # 2. 새 PDF 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            st.session_state.temp_pdf_path = tmp.name
            logging.info(f"임시 PDF 파일 생성 성공: {st.session_state.temp_pdf_path}")
        
        # 3. 세션 상태 리셋
        SessionManager.reset_for_new_file(uploaded_file)
        
        # 4. 초기 메시지 추가
        SessionManager.add_message(
            "assistant", (
                f"📂 새 PDF 파일 '{uploaded_file.name}'이(가) 업로드되었습니다.\n"
                "잠시만 기다려주세요."
                )
        )
        
        # 5. 한 번만 리런
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

def process_thought_stream(chunk: str, thought_response: str) -> tuple[str, str, bool]:
    """생각 과정 스트림 처리"""
    if "</think>" in chunk:
        parts = chunk.split("</think>", 1)
        thought_part = parts[0]
        answer_part = parts[1]
        
        thought_response += thought_part
        cleaned_thought = thought_response.replace("<think>", "").strip()
        
        return cleaned_thought, answer_part, False
    return "", "", True

def process_chat_response(qa_chain, user_input, chat_container):
    """채팅 응답 처리"""
    with chat_container:
        with st.chat_message("assistant"):
            thought_expander = st.expander("🤔 생각 과정", expanded=False)
            message_placeholder = st.empty()
            message_placeholder.write("▌")

            full_response = ""
            thought_response = ""
            processing_thought = True

            try:
                logging.info("답변 생성 시작...")
                stream = qa_chain.stream({"input": user_input})
                
                for chunk in stream:
                    answer_part = chunk.get("answer", "")
                    if not answer_part:
                        continue

                    if processing_thought:
                        cleaned_thought, remaining_answer, processing_thought = process_thought_stream(
                            answer_part, thought_response
                        )
                        
                        if cleaned_thought:
                            thought_expander.markdown(cleaned_thought)
                            thought_response = cleaned_thought
                        
                        if not processing_thought:
                            full_response = remaining_answer
                            if full_response:
                                message_placeholder.write(full_response + "▌")
                        else:
                            thought_response += answer_part
                    else:
                        full_response += answer_part
                        message_placeholder.write(full_response + "▌")

                # 최종 응답 처리
                if processing_thought:
                    full_response = thought_response.replace("<think>", "").strip()
                message_placeholder.write(full_response)

            except Exception as e:
                logging.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
                error_message = f"❌ 답변 생성 중 오류가 발생했습니다: {e}"
                message_placeholder.error(error_message)
                full_response = error_message

            return full_response

def main():
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ Settings")
        try:
            models = get_ollama_models()
            last_model = st.session_state.get("last_selected_model")
            current_model_index = models.index(last_model) if last_model and last_model in models else 0
            selected_model = st.selectbox(
                "Select an Ollama model",
                models,
                index=current_model_index,
                key="model_selector"
            ) if models else st.text("Failed to load Ollama models.")
            
            if selected_model:
                handle_model_change(selected_model)
                
        except Exception as e:
            st.error(f"Failed to load Ollama models: {e}")
            st.warning("Ollama가 설치되어 있는지, Ollama 서버가 실행 중인지 확인해주세요.")
            selected_model = None

        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        # PDF 뷰어 설정
        st.divider()
        resolution_boost = st.slider("Resolution boost", 1, 10, 1)
        width = st.slider("PDF width", 100, 1000, 1000)
        height = st.slider("PDF height", -1, 10000, 1000)

    # 레이아웃 설정
    col_left, col_right = st.columns([1, 1])

    # 오른쪽 컬럼: PDF 미리보기
    with col_right:
        st.subheader("📄 PDF Preview")
        handle_pdf_upload(uploaded_file)
        
        if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
            try:
                pdf_viewer(
                    input=st.session_state.temp_pdf_path,
                    width=width,
                    height=height,
                    key=f'pdf_viewer_{os.path.basename(st.session_state.temp_pdf_path)}',
                    resolution_boost=resolution_boost
                )
            except Exception as e:
                st.error(f"PDF 미리보기 중 오류 발생: {e}")
        elif uploaded_file:
            st.warning("PDF 미리보기를 표시할 수 없습니다.")

    # 왼쪽 컬럼: 채팅 및 설정
    with col_left:
        st.subheader("💬 Chat")
        
        # 채팅 컨테이너
        chat_container = st.container(height=500, border=True)
        
        # 채팅 메시지 표시
        with chat_container:
            if "messages" in st.session_state:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        
        # PDF 처리 관련 로직
        if not st.session_state.get("pdf_processed"):
            handle_pdf_processing(uploaded_file)

        # 채팅 입력 UI
        user_input = st.chat_input(
            "PDF 내용에 대해 질문해보세요.",
            key='user_input',
            disabled=not SessionManager.is_ready_for_chat()
        )

        # 새 메시지 처리
        if user_input:
            SessionManager.add_message("user", user_input)
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
                
            # QA 체인 검증 및 응답 생성
            qa_chain = st.session_state.get("qa_chain")
            if not qa_chain:
                error_message = "❌ QA 시스템이 준비되지 않았습니다. 모델 변경이 진행 중이거나 PDF 처리가 필요할 수 있습니다."
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                SessionManager.add_message("assistant", error_message)
            else:
                try:
                    response = process_chat_response(qa_chain, user_input, chat_container)
                    SessionManager.add_message("assistant", response)
                except Exception as e:
                    error_message = f"❌ 응답 생성 중 오류가 발생했습니다: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    SessionManager.add_message("assistant", error_message)
                    logging.error(f"응답 생성 오류: {e}", exc_info=True)

if __name__ == "__main__":
    main()