import torch
torch.classes.__path__ = [] # 호환성 문제 해결을 위한 임시 조치
import tempfile
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import logging
import json
import re
import html
from utils import (
    SessionManager,
    get_ollama_models,
    load_llm,
    process_pdf,
    update_qa_chain as util_update_qa_chain, # utils.py의 update_qa_chain 사용
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
                st.session_state.qa_chain = util_update_qa_chain( # utils.py의 함수 사용
                    st.session_state.llm,
                    st.session_state.vector_store
                )
                logging.info(f"'{selected_model}' 모델로 QA 체인 업데이트 완료.")
                # 모델 변경 완료 메시지 추가
                success_message = f"✅ '{selected_model}' 모델로 변경이 완료되었습니다."
                SessionManager.add_message("assistant", success_message)
                st.session_state.last_model_change_message = success_message
        else:
            raise ValueError("벡터 저장소 또는 LLM을 찾을 수 없습니다. PDF 재처리가 필요할 수 있습니다.")

    except Exception as e:
        error_msg = f"모델 변경 중 오류 발생: {e}"
        logging.error(f"{error_msg} ({selected_model})", exc_info=True)
        SessionManager.reset_session_state(["llm", "qa_chain"]) # pdf_processed는 유지
        SessionManager.add_message("assistant", f"❌ {error_msg}")
        st.session_state.last_model_change_message = f"❌ {error_msg}"
        
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
                f"📂 새 PDF 파일 '{uploaded_file.name}'이(가) 업로드되었습니다.\n\n"
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

def _parse_llm_output(full_llm_output: str) -> tuple[str, str, dict | None]:
    """LLM의 전체 출력을 생각 과정과 JSON 부분으로 분리하고 JSON을 파싱합니다."""
    thought_content = ""
    json_to_parse = full_llm_output.strip()

    if json_to_parse.startswith("<think>"):
        think_end_tag = "</think>"
        think_end_idx = json_to_parse.find(think_end_tag)
        if think_end_idx != -1:
            thought_content = json_to_parse[len("<think>"):think_end_idx].strip()
            json_to_parse = json_to_parse[think_end_idx + len(think_end_tag):].strip()
    
    try:
        response_data = json.loads(json_to_parse)
        return thought_content, json_to_parse, response_data
    except json.JSONDecodeError:
        logging.warning(f"JSON 파싱 실패. 원본 LLM 응답: {json_to_parse}")
        return thought_content, json_to_parse, None

def _validate_and_reformat_sources(
    answer_text: str, 
    llm_sources: list[dict], 
    session_source_docs: dict
) -> tuple[str, list[dict]]:
    """
    LLM 응답의 answer와 sources를 검증하고, 필요한 경우 출처 번호를 재정렬하며,
    툴팁에 사용될 최종 sources 정보를 구성합니다.
    """
    source_pattern = r'\[(\d+)\]'
    
    original_ids_in_answer = sorted(list(set(map(int, re.findall(source_pattern, answer_text)))))

    if not original_ids_in_answer:
        return answer_text, []

    id_map = {original_id: new_id for new_id, original_id in enumerate(original_ids_in_answer, 1)}
    
    def replace_id_in_answer(match):
        original_id = int(match.group(1))
        return f"[{id_map.get(original_id, original_id)}]"
    
    updated_answer_text = re.sub(source_pattern, replace_id_in_answer, answer_text)

    final_sources_for_tooltip = []
    llm_sources_map = {s.get("id"): s for s in llm_sources if isinstance(s.get("id"), int)}

    for original_id in original_ids_in_answer:
        new_id = id_map[original_id]
        
        source_info_from_llm = llm_sources_map.get(original_id)
        actual_doc_chunk = session_source_docs.get(str(original_id))

        text_for_tooltip = "출처 정보를 찾을 수 없습니다."
        page_for_tooltip = "N/A"

        if actual_doc_chunk and 'content' in actual_doc_chunk:
            text_for_tooltip = actual_doc_chunk['content']
            page_for_tooltip = actual_doc_chunk.get('page', 'N/A') # 이미 문자열 또는 "N/A"
            logging.info(f"툴팁 ID {new_id} (원본 LLM ID: {original_id}): 실제 문서 조각 사용 (페이지: {page_for_tooltip}).")
        elif source_info_from_llm and isinstance(source_info_from_llm.get("text"), str):
            text_for_tooltip = source_info_from_llm["text"]
            page_for_tooltip = source_info_from_llm.get("page", "N/A") # LLM이 문자열로 제공하거나 기본값 "N/A"
            logging.warning(f"툴팁 ID {new_id} (원본 LLM ID: {original_id}): 실제 문서 조각 없음. LLM 제공 text 사용 (페이지: {page_for_tooltip}).")
        else:
            logging.error(f"툴팁 ID {new_id} (원본 LLM ID: {original_id}): 실제 문서 조각 및 LLM 제공 text 모두 부적합/누락.")

        final_sources_for_tooltip.append({
            "id": new_id,
            "text": text_for_tooltip,
            "page": page_for_tooltip,
            "original_llm_id": original_id
        })
        
    final_sources_for_tooltip.sort(key=lambda s: s["id"])
            
    return updated_answer_text, final_sources_for_tooltip

def _create_tooltip_html(source_id: str, source_text: str, source_page: str) -> str:
    """개별 출처에 대한 툴팁 HTML을 생성합니다."""
    escaped_tooltip_text = html.escape(source_text)
    page_info = f" (페이지 {source_page})" if source_page and source_page != "N/A" else ""
    
    tooltip_content_html = (
        f'<div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #dee2e6;color:#666;font-size:0.8em">'
        f'참고한 문서 내용{page_info}:</div>'
        f'<div style="white-space: pre-wrap; word-wrap: break-word;">{escaped_tooltip_text}</div>'
    )
    
    return (
        f'<span style="position:relative;display:inline">'
        f'<sup style="color:#666;font-size:0.8em;cursor:help;margin-left:2px">[{source_id}]</sup>'
        f'<div style="visibility:hidden;position:absolute;z-index:1000;width:400px;background-color:#f8f9fa;'
        f'color:#333;text-align:left;border-radius:4px;padding:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1);'
        f'font-size:0.9em;line-height:1.4;left:50%;transform:translateX(-50%);top:100%;margin-top:5px;'
        f'opacity:0;transition:opacity 0.2s; max-height: 300px; overflow-y: auto;">'
        f'{tooltip_content_html}</div></span>'
    )

def process_chat_response(qa_chain, user_input, chat_container):
    """채팅 응답 처리"""
    with chat_container:
        with st.chat_message("assistant"):
            thought_expander = st.expander("🤔 생각 과정", expanded=False)
            message_placeholder = st.empty()
            message_placeholder.write("답변 생성 중...") # 초기 메시지

            try:
                logging.info("답변 생성 시작...")
                
                full_llm_output_chunks = []
                accumulated_raw_output = ""
                displayed_thought_once = False # 생각 과정이 한 번이라도 표시되었는지 추적
                
                # 1. Stream을 사용하여 LLM 응답을 실시간으로 받음
                for chunk in qa_chain.stream({"input": user_input}):
                    full_llm_output_chunks.append(chunk)
                    accumulated_raw_output = "".join(full_llm_output_chunks)

                    # "생각 과정" 동적 업데이트
                    think_start_tag = "<think>"
                    think_end_tag = "</think>"
                    start_idx = accumulated_raw_output.find(think_start_tag)

                    if start_idx != -1:
                        # 첫 번째 <think> 블록의 끝을 찾음
                        current_think_block_end_idx = accumulated_raw_output.find(think_end_tag, start_idx)
                        if current_think_block_end_idx != -1:
                            # 완전한 <think>...</think> 블록을 찾음
                            current_thought_content = accumulated_raw_output[
                                start_idx + len(think_start_tag) : current_think_block_end_idx
                            ].strip()
                            thought_expander.markdown(current_thought_content)
                            displayed_thought_once = True
                            # 생각 과정이 표시된 후 메시지 업데이트
                            if message_placeholder: # placeholder가 아직 유효한 경우
                                message_placeholder.write("생각 완료, 답변 구성 중...")
                        else:
                            # <think> 시작은 했지만 아직 끝나지 않음
                            partial_thought = accumulated_raw_output[start_idx + len(think_start_tag):].strip()
                            thought_expander.markdown(f"{partial_thought}...") # 진행 중임을 표시
                            if message_placeholder:
                                message_placeholder.write("생각 중...")
                    elif not displayed_thought_once and message_placeholder: # 아직 <think> 태그가 없거나, 이미 <think> 블록이 끝났고 JSON 부분일 때
                        message_placeholder.write("답변 수신 중...")
                
                full_llm_output = accumulated_raw_output # 최종 누적된 결과
                if not full_llm_output:
                    raise ValueError("LLM으로부터 빈 응답을 받았습니다.")

                # 2. LLM 출력 파싱 (생각 과정, raw JSON, 파싱된 데이터)
                thought_content, raw_json_part, response_data = _parse_llm_output(full_llm_output)

                # 3. 최종 생각 과정 표시 (스트리밍 중 업데이트 되었을 수 있으나, 최종 파싱 결과로 한 번 더 업데이트)
                if thought_content:
                    thought_expander.markdown(thought_content)
                else:
                    thought_expander.empty()

                # 4. JSON 파싱 결과 확인 및 답변 처리
                if response_data is None: # JSON 파싱 실패
                    error_content = f"JSON 파싱에 실패했습니다. LLM으로부터 받은 JSON 부분: \n```json\n{html.escape(raw_json_part)}\n```"
                    message_placeholder.error(error_content)
                    SessionManager.add_message("assistant", f"JSON 파싱 실패: {raw_json_part}")
                    return # 여기서 함수 종료

                answer = response_data.get("answer", "")
                llm_provided_sources = response_data.get("sources", [])

                if not answer:
                    message_placeholder.markdown("LLM으로부터 답변 내용을 받지 못했습니다.")
                    SessionManager.add_message("assistant", "LLM으로부터 답변 내용을 받지 못했습니다.")
                    return

                session_source_documents = st.session_state.get("source_documents", {})
                
                processed_answer, tooltip_sources = _validate_and_reformat_sources(
                    answer, 
                    llm_provided_sources, 
                    session_source_documents
                )
                
                tooltip_sources_map = {str(s["id"]): s for s in tooltip_sources}

                source_pattern = r'\[(\d+)\]'

                def replace_with_tooltip(match):
                    source_id_str = match.group(1)
                    source_data = tooltip_sources_map.get(source_id_str)
                    if source_data:
                        return _create_tooltip_html(
                            source_id_str, 
                            source_data["text"], 
                            source_data["page"]
                        )
                    return match.group(0)

                formatted_answer_with_tooltips = re.sub(source_pattern, replace_with_tooltip, processed_answer)
                
                tooltip_style = '<style>span:hover div{visibility:visible !important;opacity:1 !important}</style>'
                final_html_output = tooltip_style + formatted_answer_with_tooltips
                
                message_placeholder.markdown(final_html_output, unsafe_allow_html=True)
                SessionManager.add_message("assistant", final_html_output)

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
        selected_model = None  # 항상 초기화
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

        # 모델 변경 시 처리
        # 세션 상태에 last_selected_model이 없거나 선택된 모델과 다를 경우
        if (
            selected_model
            and selected_model != st.session_state.get("last_selected_model")
        ):
            handle_model_change(selected_model)

        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        # PDF 뷰어 설정
        st.divider()
        resolution_boost = st.slider("Resolution boost", 1, 10, 1)
        width = st.slider("PDF width", 100, 1000, 1000)
        height = st.slider("PDF height", 100, 10000, 1000) # 최소 높이를 100으로 변경

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
        # JavaScript 기반 position:fixed 툴팁 사용 시 위 CSS 오버라이드는 불필요할 수 있습니다.
        
        # 채팅 메시지 표시
        display_chat_messages(chat_container)

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
                    process_chat_response(qa_chain, user_input, chat_container)
                except Exception as e:
                    error_message = f"❌ 응답 생성 중 오류가 발생했습니다: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    SessionManager.add_message("assistant", error_message)
                    logging.error(f"응답 생성 오류: {e}", exc_info=True)

if __name__ == "__main__":
    main()