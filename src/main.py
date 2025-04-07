import torch
torch.classes.__path__ = []
import tempfile
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
import logging
from utils import (
    get_ollama_models,
    load_pdf_docs,
    get_embedder,
    split_documents,
    create_vector_store,
    init_llm,
)

# 페이지 설정
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with Ollama LLM")

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 사이드바 설정: 모델 선택 및 PDF 업로드
with st.sidebar:
    st.header("Settings")
    try:
        models = get_ollama_models()
        selected_model = st.selectbox("Select an Ollama model", models) if models else st.text("Failed to load Ollama models.")
    except Exception as e:
        st.error(f"Failed to load Ollama models: {e}")
        st.warning("Ollama가 설치되어 있는지, Ollama 서버가 실행 중인지 확인해주세요.")
        selected_model = None
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    resolution_boost = st.slider(label="Resolution boost", min_value=1, max_value=10, value=1)
    width = st.slider(label="PDF width", min_value=100, max_value=1000, value=1000)
    height = st.slider(label="PDF height", min_value=-1, max_value=10000, value=1000)

# 2열 레이아웃: 왼쪽에는 설정 및 대화, 오른쪽에는 PDF 미리보기
col_left, col_right = st.columns([1, 1])

# PDF 미리보기
with col_right:
    st.header("📄 PDF Preview")
    if uploaded_file:
        try:
            # 임시 파일에 업로드된 PDF 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            # 임시 파일 경로를 사용하여 pdf_viewer 호출
            pdf_viewer(input=tmp_path, width=width, height=height, key=f'pdf_viewer_{uploaded_file.name}', resolution_boost=resolution_boost)
        except Exception as e:
            st.error(f"PDF 미리보기 중 오류 발생: {e}")

# 대화 및 설정
with col_left:
    st.header("💬 Chat")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_selected_model" not in st.session_state:
        st.session_state.last_selected_model = None
        
    # 대화 메시지를 저장할 컨테이너 생성
    chat_container = st.container(height=500, border=True, key="chat_container")

    # 대화 메시지 표시
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # 모델 변경 시 업데이트
    if selected_model and selected_model != st.session_state.get("last_selected_model"):
        st.session_state.last_selected_model = selected_model
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"🛠️ 모델 {selected_model}이(가) 선택되었습니다."
        })
        with chat_container:
            with st.chat_message("assistant"):
                st.write(f"🛠️ 모델 {selected_model}이(가) 선택되었습니다.")
    
    # PDF 파일 업로드 시 세션 상태 초기화
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        
        # 파일이 변경되었을 때만 세션 상태 초기화
        if st.session_state.get("last_uploaded_file") != file_bytes:
            st.session_state.last_uploaded_file = file_bytes
            
            # 관련 캐시 및 상태 초기화
            load_pdf_docs.clear()
            get_embedder.clear()
            split_documents.clear()
            create_vector_store.clear()
            
            # 세션 상태 초기화
            for key in ["pdf_processed", "pdf_completed", "qa_chain", "vector_store", "llm", "pdf_processing", "pdf_message_logged"]:
                st.session_state.pop(key, None)
    
    # PDF 문서 처리
    if uploaded_file and not st.session_state.get("pdf_completed", False):
        # PDF 파일이 업로드되었고 처리되지 않은 경우
        if not st.session_state.get("pdf_message_logged", False):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📂 PDF 파일 {uploaded_file.name}이(가) 업로드되었습니다."
            })
            st.session_state.pdf_message_logged = True
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(f"📂 PDF 파일 {uploaded_file.name}이(가) 업로드되었습니다.")

        # PDF 문서 처리 중 메시지 출력
        with chat_container:
            with st.spinner("📄 문서를 처리하는 중... 잠시만 기다려 주세요.", show_time=True):
                try:
                    docs = load_pdf_docs(uploaded_file.getvalue())        
                    embedder = get_embedder(model_name="BAAI/bge-m3",
                                            model_kwargs={'device': 'cuda'},
                                            encode_kwargs={'device': 'cuda'})
                    documents = split_documents(docs, embedder)
                    vector_store = create_vector_store(documents, embedder)
                    
                    if isinstance(selected_model, str):
                        llm = init_llm(selected_model)
                    else:
                        raise ValueError("❌ LLM 초기화에 실패했습니다.")
                except ValueError as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": str(e)
                    })
                
                # 문서 처리 완료 후 세션 상태 업데이트
                st.session_state.vector_store = vector_store
                st.session_state.llm = llm  

                # QA 체인 생성
                QA_PROMPT = ChatPromptTemplate.from_messages([
                    # 시스템 메시지 (지침)
                    ("system", 
                    """다음의 문맥에 기반해서만 사용자 질문에 답변하세요:

                    <context>
                    {context}
                    </context>"""),
                        
                    # 이전 채팅 기록 자리
                    MessagesPlaceholder(variable_name="chat_history"),
                    
                    # 현재 사용자 입력
                    ("human", "{input}")
                    ])
                
                # 문서 체인 생성
                combine_chain = create_stuff_documents_chain(llm, QA_PROMPT)
                qa_chain = create_retrieval_chain(vector_store.as_retriever(search_type="similarity",
                                                                            search_kwargs={"k": 20}), combine_chain)

                st.session_state.qa_chain = qa_chain
                st.session_state.pdf_completed = True  

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"✅ PDF 파일 {uploaded_file.name}의 문서 처리가 완료되었습니다."
                })
                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(f"✅ PDF 파일 {uploaded_file.name}의 문서 처리가 완료되었습니다.")
                        
        # 예시 질문 생성
        with chat_container:
            with st.spinner("📄 다음 문서에 대한 대표 질문 5가지를 생성 중... 잠시만 기다려 주세요.", show_time=True):
                try:
                    suggestion_prompt = "다음 문서에 기반해서 사용자가 할 수 있는 대표적인 질문 5가지를 제시해 주세요:\n\n"
                    doc_sample_text = "\n\n".join([d.page_content for d in documents[:3]])[:2000]  # 너무 길지 않게 자르기
                    suggestion_input = suggestion_prompt + doc_sample_text

                    suggestion_response = llm.invoke(suggestion_input)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"💡 다음은 문서를 기반으로 할 수 있는 질문 예시입니다:\n\n{suggestion_response}"
                    })
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.write(f"💡 다음은 문서를 기반으로 할 수 있는 질문 예시입니다:\n\n{suggestion_response}")
                except Exception as e:
                    st.warning(f"예시 질문 생성 중 오류 발생: {e}")
    
    # 메시지 입력
    user_input = st.chat_input("메시지를 입력하세요")
    
    # 사용자 입력 처리
    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })

        # QA 체인 초기화 확인
        qa_chain = st.session_state.qa_chain
        if not qa_chain:
            with chat_container:
                with st.chat_message("assistant"):
                    st.write("❌ 문서 처리가 완료되지 않았습니다. PDF를 먼저 업로드하세요.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "❌ 문서 처리가 완료되지 않았습니다. PDF를 먼저 업로드하세요."
            })
            st.stop()
            
        # 답변 생성
        with chat_container:
            with st.chat_message("assistant"):
                status_message = st.empty()
                status_message.write("🤖 답변 생성 중...")
                message_placeholder = st.empty()
                answer = ""
                try:
                    # 스트리밍 응답 처리
                    for chunk in qa_chain.stream({"input": user_input, "chat_history": []}):
                        if "answer" in chunk:
                            if not answer:
                                status_message.empty()
                            answer += chunk["answer"]
                            message_placeholder.write(answer + "▌")
                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
                    message_placeholder.write(answer)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer
        })