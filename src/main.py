import torch
torch.classes.__path__ = [] # 호환성 문제 해결을 위한 임시 조치
import tempfile
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import logging
from utils import (
    init_session_state,
    reset_session_state,
    get_ollama_models,
    load_llm,
    QA_PROMPT,
    process_pdf,
)

# 페이지 설정
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
    )

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )

# 세션 상태 초기화
init_session_state()

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ Settings")
    try:
        models = get_ollama_models()
        # 세션 상태에 저장된 마지막 모델 확인 및 인덱스 설정
        last_model = st.session_state.get("last_selected_model")
        current_model_index = models.index(last_model) if last_model and last_model in models else 0
        selected_model = st.selectbox(
            "Select an Ollama model",
            models,
            index=current_model_index,
            key="model_selector" # 위젯 상태 유지를 위한 키 추가
        ) if models else st.text("Failed to load Ollama models.")
    except Exception as e:
        st.error(f"Failed to load Ollama models: {e}")
        st.warning("Ollama가 설치되어 있는지, Ollama 서버가 실행 중인지 확인해주세요.")
        selected_model = None

    # --- 모델 변경 감지 및 처리 로직 ---
    if selected_model and selected_model != st.session_state.get("last_selected_model"):
        old_model = st.session_state.get("last_selected_model", "N/A")
        st.session_state.last_selected_model = selected_model
        st.session_state.llm = None # 이전 LLM 인스턴스 상태 제거
        st.session_state.qa_chain = None # 이전 QA 체인 상태 제거
        # load_llm 캐시를 직접 지울 필요는 보통 없음 (st.cache_resource가 인자 기반 캐싱)
        # load_llm.clear() # 필요 시 주석 해제

        logging.info(f"LLM 변경 감지: {old_model} -> {selected_model}")

        if st.session_state.get("pdf_processed"):
            # PDF가 이미 처리된 경우, 새 모델로 LLM 및 QA 체인 재생성
            st.info(f"모델을 '{selected_model}'(으)로 변경하는 중...")
            try:
                # 1. 새 LLM 로드 및 저장
                with st.spinner(f"'{selected_model}' 모델 로딩 중..."):
                    # utils에서 load_llm 함수 사용 (캐시 활용)
                    new_llm = load_llm(selected_model)
                    st.session_state.llm = new_llm

                # 2. QA 체인 재생성 (기존 벡터 저장소 사용)
                if st.session_state.get("vector_store") and st.session_state.get("llm"):
                    with st.spinner("QA 시스템 업데이트 중..."):
                        # utils에서 임포트한 QA_PROMPT 사용
                        from langchain.chains.combine_documents import create_stuff_documents_chain
                        from langchain.chains.retrieval import create_retrieval_chain

                        combine_chain = create_stuff_documents_chain(st.session_state.llm, QA_PROMPT)
                        retriever = st.session_state.vector_store.as_retriever(
                            search_type="mmr",
                            search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.7},
                        )
                        new_qa_chain = create_retrieval_chain(retriever, combine_chain)
                        st.session_state.qa_chain = new_qa_chain
                        logging.info(f"'{selected_model}' 모델로 QA 체인 업데이트 완료.")
                        st.success(f"모델이 '{selected_model}'(으)로 성공적으로 변경되었습니다.")
                        # 채팅에 변경 완료 메시지 추가
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"✅ 모델이 '{selected_model}'(으)로 변경되었습니다. 이제 새 모델로 질문할 수 있습니다."
                        })
                        st.rerun() # 변경사항 즉시 반영 (채팅 메시지 표시 등)
                else:
                    st.error("QA 시스템 업데이트 실패: 벡터 저장소 또는 LLM을 찾을 수 없습니다.")
                    logging.error("QA 시스템 업데이트 실패: 벡터 저장소 또는 LLM 상태 없음")
                    st.session_state.pdf_processed = False # 처리 실패 상태로 변경
                    st.session_state.qa_chain = None

            except Exception as e:
                st.error(f"모델 변경 중 오류 발생: {e}")
                logging.error(f"모델 변경 중 오류 ({selected_model}): {e}", exc_info=True)
                st.session_state.llm = None
                st.session_state.qa_chain = None
                st.session_state.pdf_processed = False # 오류 시 처리 실패 상태로 변경
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 모델을 '{selected_model}'(으)로 변경하는 중 오류가 발생했습니다: {e}"
                })
                st.rerun() # 오류 메시지 표시를 위해 rerun

        else:
            # PDF가 아직 처리되지 않은 경우, 로그만 남김 (사용자에게는 파일 업로드 시 반영됨)
            logging.info(f"모델 선택 변경됨: {selected_model}. PDF 업로드 시 적용됩니다.")
            # 필요 시 채팅에 정보 메시지 추가 가능
            # st.session_state.messages.append({"role": "assistant", "content": f"ℹ️ 모델이 '{selected_model}'(으)로 선택되었습니다."})
            # if message added, uncomment below
            # st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # PDF 뷰어 설정
    st.divider()
    resolution_boost = st.slider(label="Resolution boost", min_value=1, max_value=10, value=1)
    width = st.slider(label="PDF width", min_value=100, max_value=1000, value=1000)
    height = st.slider(label="PDF height", min_value=-1, max_value=10000, value=1000)

# 레이아웃 설정
col_left, col_right = st.columns([1, 1])

# 오른쪽 컬럼: PDF 미리보기
with col_right:
    st.subheader("📄 PDF Preview")
    # PDF 미리보기
    if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_file_name"):
        if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
            try:
                os.remove(st.session_state.temp_pdf_path)
                logging.info("이전 임시 PDF 파일 삭제 성공")
            except Exception as e:
                logging.warning(f"이전 임시 PDF 파일 삭제 실패: {e}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state.temp_pdf_path = tmp.name
                logging.info(f"임시 PDF 파일 생성 성공: {st.session_state.temp_pdf_path}")
        except Exception as e:
            st.error(f"임시 PDF 파일 생성 실패: {e}")
            st.session_state.temp_pdf_path = None

    # PDF 미리보기 표시
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
    chat_container = st.container(height=500, border=True)

    new_file_uploaded = uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_file_name")
    if new_file_uploaded:
        if st.session_state.temp_pdf_path:
            reset_session_state(uploaded_file)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📂 새 PDF 파일 '{uploaded_file.name}'이(가) 업로드되었습니다."
                })
        else:
            st.warning("PDF 파일을 임시로 저장하는 데 실패했습니다. 다시 시도해 주세요.")
            
    # PDF 처리 상태 확인
    if uploaded_file and st.session_state.temp_pdf_path and not st.session_state.get("pdf_processed") and not st.session_state.get("pdf_processing_error"):
        # 처리 시작 전, 현재 선택된 모델 확인
        current_selected_model = st.session_state.get("last_selected_model")
        if not current_selected_model:
             st.warning("모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
        else:
            with chat_container:
                with st.spinner(f"📄 PDF 문서 처리 중... 잠시만 기다려 주세요."):
                    # process_pdf 호출 시 현재 세션의 모델 사용
                    process_pdf(
                        uploaded_file,
                        current_selected_model, # 세션 상태의 최신 모델 사용
                        st.session_state.temp_pdf_path
                    )
                    # process_pdf 내부에서 session_state를 업데이트하므로 반환값 저장 불필요
            
    # 채팅 메시지 표시
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 채팅 입력창
    # 입력 가능 조건: PDF 처리 완료 + 오류 없음 + QA 체인 존재
    is_ready_for_input = st.session_state.get("pdf_processed") and not st.session_state.get("pdf_processing_error") and st.session_state.get("qa_chain") is not None
    user_input = st.chat_input(
        "PDF 내용에 대해 질문해보세요.",
        key='user_input',
        disabled=not is_ready_for_input
    )

    # 사용자 입력 처리
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            })
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)

        # 답변 생성
        # 답변 생성 전 QA 체인 유효성 확인 (모델 변경 중 None일 수 있음)
        qa_chain = st.session_state.get("qa_chain")
        if not qa_chain:
            error_message = "❌ QA 시스템이 준비되지 않았습니다. 모델 변경이 진행 중이거나 PDF 처리가 필요할 수 있습니다."
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                })
            # 채팅창에 오류 메시지 즉시 표시
            with chat_container:
                with st.chat_message("assistant"):
                    st.error(error_message) # 오류 강조 표시
            # st.rerun() # 필요 시 rerun으로 즉시 업데이트
        else:
            # QA 체인이 준비된 경우 답변 생성 진행
            with chat_container:
                with st.chat_message("assistant"):
                    # 생각 과정 expander를 미리 생성하고 접어둡니다.
                    thought_expander = st.expander("🤔 생각 과정", expanded=False)
                    message_placeholder = st.empty()
                    message_placeholder.write("▌")

                    full_response = "" # 최종 사용자 대상 답변용
                    thought_response = "" # 생각 과정 누적용
                    processing_thought = True # 생각 과정 처리 여부

                    try:
                        # 답변 생성
                        logging.info("답변 생성 시작...")
                        stream = qa_chain.stream({"input": user_input})
                        for chunk in stream:
                            answer_part = chunk.get("answer", "")
                            if not answer_part:
                                continue

                            if processing_thought:
                                if "</think>" in answer_part:
                                    # 이 청크에서 생각 과정의 끝을 찾음
                                    parts = answer_part.split("</think>", 1)
                                    thought_part = parts[0]
                                    answer_part_after_think = parts[1]

                                    # 생각의 마지막 부분을 누적
                                    thought_response += thought_part

                                    # 생각을 익스팬더에 표시
                                    cleaned_thought = thought_response.replace("<think>", "").strip()
                                    if cleaned_thought:
                                        # 미리 생성된 expander 내부에 markdown으로 내용을 채웁니다.
                                        thought_expander.markdown(cleaned_thought)

                                    processing_thought = False # 실제 답변 처리로 전환

                                    # 청크의 나머지 부분부터 실제 답변 누적 시작
                                    full_response += answer_part_after_think
                                    if full_response: # 내용이 있을 때만 플레이스홀더 업데이트
                                        message_placeholder.write(full_response + "▌")
                                else:
                                    # Still accumulating the thought part
                                    thought_response += answer_part
                            else:
                                # Accumulating the answer part after </think>
                                full_response += answer_part
                                message_placeholder.write(full_response + "▌")

                        # 루프 종료 후 최종 업데이트
                        if processing_thought:
                            # </think>를 찾지 못함, 누적된 전체 생각을 답변으로 처리
                            cleaned_thought = thought_response.replace("<think>", "").strip()
                            message_placeholder.write(cleaned_thought)
                            full_response = cleaned_thought # 상태 저장을 위해 할당
                        else:
                            # 생각과 답변 처리 후 정상 종료
                            message_placeholder.write(full_response) # 커서 없이 최종 답변 작성

                    except Exception as e:
                        logging.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
                        error_message = f"❌ 답변 생성 중 오류가 발생했습니다: {e}"
                        message_placeholder.error(error_message)
                        full_response = error_message # 상태 저장을 위해 full_response를 오류 메시지로 설정

            # 최종 *답변* 부분 (또는 오류)을 세션 상태에 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                })