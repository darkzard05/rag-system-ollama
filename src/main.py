import torch
torch.classes.__path__ = []
import tempfile
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
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

logging.basicConfig(level=logging.INFO)

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    # 모델 목록 가져오기
    models = get_ollama_models()
    selected_model = st.selectbox("사용할 Ollama 모델 선택", models) if models else st.text("Ollama 모델을 불러올 수 없습니다.")
    uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")
    resolution_boost = st.slider(label="Resolution boost", min_value=1, max_value=10, value=1)
    width = st.slider(label="PDF width", min_value=100, max_value=1000, value=1000)
    height = st.slider(label="PDF height", min_value=-1, max_value=10000, value=1000)

user_input = st.chat_input("메시지를 입력하세요")

# 2열 레이아웃: 왼쪽에는 설정 및 대화, 오른쪽에는 PDF 미리보기
col_left, col_right = st.columns([1, 1])

with col_right:
    st.header("📄 PDF 미리보기")
    if uploaded_file:
        st.subheader(f"파일명: {uploaded_file.name}")
        try:
            # 임시 파일에 업로드된 PDF 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            # 임시 파일 경로를 사용하여 pdf_viewer 호출
            pdf_viewer(input=tmp_path, height=height, width=width, resolution_boost=resolution_boost)
        except Exception as e:
            st.error(f"PDF 미리보기 중 오류 발생: {e}")

with col_left:
    st.header("💬 대화")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_selected_model" not in st.session_state:
        st.session_state.last_selected_model = None

    # 상단에 저장된 대화 메시지를 순서대로 출력
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
        st.rerun()
    
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
            for key in ["pdf_processed", "pdf_completed", "qa_chain", "vector_store", "llm", "pdf_processing", "pdf_message_logged"]:
                st.session_state.pop(key, None)
            st.rerun()
    
    # PDF 문서 처리
    if uploaded_file and not st.session_state.get("pdf_completed", False):
        if not st.session_state.get("pdf_message_logged", False):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📂 PDF 파일 {uploaded_file.name}이(가) 업로드되었습니다. 문서를 처리합니다..."
            })
            st.session_state.pdf_message_logged = True

        with st.spinner("📄 문서를 처리하는 중... 잠시만 기다려 주세요."):
            docs = load_pdf_docs(uploaded_file.getvalue())
            if not docs:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "❌ 문서 처리에 실패했습니다."
                })
                st.stop()
            
            embedder = get_embedder(model_name="BAAI/bge-m3",
                                    model_kwargs={'device': 'cuda'},
                                    encode_kwargs={'device': 'cuda'})
            if embedder is None:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "❌ 임베더 초기화에 실패했습니다."
                })
                st.stop()
            
            try:
                documents = split_documents(docs, embedder)
                if not documents:
                    raise ValueError("❌ 문서 분할에 실패했습니다.")
                vector_store = create_vector_store(documents, embedder)
                if not vector_store:
                    raise ValueError("❌ 벡터 저장소 생성에 실패했습니다.")
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"오류가 발생했습니다: {e}"
                })
                st.stop()
            
            st.session_state.vector_store = vector_store

            if isinstance(selected_model, str):
                llm = init_llm(selected_model)
            else:
                llm = None
            if llm is None:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "❌ LLM 초기화에 실패했습니다."
                })
                st.stop()
                
            st.session_state.llm = llm  

            QA_PROMPT = PromptTemplate.from_template(
            """
            당신은 문서 분석 및 요약 전문가입니다.
            아래 제공된 문서 컨텍스트 내의 정보만 활용하여, 주어진 질문에 대해 정확하고 명확하게 답변하십시오.
            불확실하거나 확인되지 않은 내용은 언급하지 마시고, 항상 한국어로 응답하십시오.

            [컨텍스트]
            {context}
            
            [질문]
            {input}

            [답변]
            """
            )
            combine_chain = create_stuff_documents_chain(llm, QA_PROMPT)
            qa_chain = create_retrieval_chain(vector_store.as_retriever(search_type="similarity",
                                                                        search_kwargs={"k": 20}), combine_chain)

            st.session_state.qa_chain = qa_chain
            st.session_state.pdf_completed = True  

            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"✅ PDF 파일 {uploaded_file.name}의 문서 처리가 완료되었습니다."
            })
            st.rerun()
    
    # 사용자 입력 처리
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })

        qa_chain = st.session_state.get("qa_chain")
        if not qa_chain:
            with st.chat_message("assistant"):
                st.write("❌ 문서 처리가 완료되지 않았습니다. PDF를 먼저 업로드하세요.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "❌ 문서 처리가 완료되지 않았습니다. PDF를 먼저 업로드하세요."
            })
            st.stop()

        with st.chat_message("assistant"):
            with st.spinner("🤖 답변 생성 중..."):
                try:
                    response = st.session_state.qa_chain.invoke({"input": user_input})
                    answer = response["answer"]
                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer
        })