import os
import torch
torch.classes.__path__ = []
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Ollama 모델 목록 가져오기
def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.split("\n")[1:] if line]
        return models
    except Exception as e:
        logging.error(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}")
        return []

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with Ollama LLM")

logging.basicConfig(level=logging.DEBUG)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_selected_model" not in st.session_state:
    st.session_state.last_selected_model = None

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    models = get_ollama_models()
    selected_model = st.selectbox("사용할 Ollama 모델 선택", models) if models else st.text("Ollama 모델을 불러올 수 없습니다.")
    uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")

# 상단에 저장된 대화 메시지를 순서대로 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 모델 변경 시 session_state에 메시지 추가 (출력은 위 for문에서 한 번에 됨)
if selected_model and selected_model != st.session_state.get("last_selected_model"):
    st.session_state.last_selected_model = selected_model  # 선택한 모델 저장
    new_message = {
        "role": "assistant",
        "content": f"🛠️ 모델 `{selected_model}`이(가) 선택되었습니다."
    }
    st.session_state.messages.append(new_message)
    st.rerun()  # 업데이트 후 재실행하여 전체 대화 기록 출력

@st.cache_data(show_spinner=False)
def load_pdf_docs(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        os.remove(temp_path)
        return docs
    except Exception as e:
        logging.error(f"PDF 로드 중 오류 발생: {e}")
        return []

@st.cache_resource(show_spinner=False)
def get_embedder(model_name="intfloat/e5-base-v2", model_kwargs={'device': 'cuda'}):
    return HuggingFaceEmbeddings(model_name=model_name,
                                 model_kwargs=model_kwargs)

@st.cache_data(show_spinner=False)
def split_documents(_docs, _embedder):
    try:
        chunker = SemanticChunker(_embedder)
        return chunker.split_documents(_docs)
    except Exception as e:
        logging.error(f"문서 분할 중 오류 발생: {e}")
        return []

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents, _embedder):
    try:
        return FAISS.from_documents(_documents, _embedder)
    except Exception as e:
        logging.error(f"벡터 저장소 생성 중 오류 발생: {e}")
        return None

@st.cache_resource(show_spinner=False)
def init_llm(model_name):
    try:
        return OllamaLLM(model=model_name, device='cuda')
    except Exception as e:
        logging.error(f"LLM 초기화 중 오류 발생: {e}")
        return None

# PDF 파일 업로드 시 기존 상태 초기화 및 처리
if uploaded_file:
    file_bytes = uploaded_file.getvalue()  # 파일 내용 가져오기

    # ✅ 새로운 PDF 업로드 감지 후, 한 번만 상태 초기화
    if st.session_state.get("last_uploaded_file") != file_bytes:
        st.session_state.last_uploaded_file = file_bytes  # 파일 내용을 기준으로 비교

        # ✅ 캐시 무효화
        load_pdf_docs.clear()
        get_embedder.clear()
        split_documents.clear()
        create_vector_store.clear()

        # ✅ 🔥 세션 상태 초기화
        for key in ["pdf_processed", "pdf_completed", "qa_chain", "vector_store", "llm", "pdf_processing", "pdf_message_logged"]:
            st.session_state.pop(key, None)  # 존재하는 경우만 삭제

        # ✅ UI 즉시 갱신하여 새로운 PDF 감지 (이후 추가적인 rerun 없음)
        st.rerun()

# ✅ "문서 처리 중" 메시지가 한 번만 출력되도록 조절
if uploaded_file and not st.session_state.get("pdf_completed", False):
    if not st.session_state.get("pdf_message_logged", False):  
        st.session_state.messages.append({
            'role': 'assistant',
            'content': f'📂 PDF 파일 `{uploaded_file.name}`이(가) 업로드되었습니다. 문서를 처리합니다...'
        })
        st.session_state.pdf_message_logged = True  # 메시지 중복 추가 방지

    # 🔥 문서 처리 과정에서 깜박임 방지를 위해 st.spinner() 사용
    with st.spinner("📄 문서를 처리하는 중... 잠시만 기다려 주세요."):
        # 1️⃣ 문서 로딩
        docs = load_pdf_docs(uploaded_file.getvalue())
        if not docs:
            st.session_state.messages.append({
                'role': 'assistant', 
                'content': '❌ 문서 처리에 실패했습니다.'
            })
            st.stop()

        # 1️⃣ 문서 임베딩 생성
        embedder = get_embedder(model_name="intfloat/e5-base-v2",
                                model_kwargs={'device': 'cuda'})
        
        # 2️⃣ 문서 분할
        with ThreadPoolExecutor() as executor:
            future_split = executor.submit(split_documents, docs, embedder)
            documents = future_split.result()

        # 3️⃣ 벡터 저장소 생성
        vector_store = create_vector_store(documents, embedder)
        if vector_store is None:
            st.session_state.messages.append({
                'role': 'assistant', 
                'content': '❌ 벡터 저장소 생성에 실패했습니다.'
            })
            st.stop()
        st.session_state.vector_store = vector_store  

        # 4️⃣ LLM 초기화
        if isinstance(selected_model, str):
            llm = init_llm(selected_model)
        else:
            llm = None
        if llm is None:
            st.session_state.messages.append({
                'role': 'assistant', 
                'content': '❌ LLM 초기화에 실패했습니다.'
            })
            st.stop()
        st.session_state.llm = llm  

        # 5️⃣ QA 체인 생성
        prompt = """
        당신은 문서 분석 및 요약 전문가입니다.
        아래 제공된 문서 컨텍스트 내의 정보만 활용하여, 주어진 질문에 대해 정확하고 명확하게 답변하십시오.
        불확실하거나 확인되지 않은 내용은 언급하지 마시고, 항상 한국어로 응답하십시오.

        [컨텍스트]
        {context}
        
        [질문]
        {input}

        [답변]
        """
        QA_PROMPT = PromptTemplate.from_template(prompt)
        combine_chain = create_stuff_documents_chain(llm, QA_PROMPT)
        qa_chain = create_retrieval_chain(vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 100}), combine_chain)

        # 6️⃣ QA 체인 저장
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_completed = True  

        # 7️⃣ 문서 처리 완료 메시지 추가 (한 번만 실행됨)
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': f'✅ PDF 파일 `{uploaded_file.name}` 문서 처리가 완료되었습니다.'
        })

        # ✅ 문서 처리가 완료된 이후, 최종적으로 한 번만 `st.rerun()`
        st.rerun()


# 사용자 입력 시 처리
user_input = st.chat_input("메시지를 입력하세요")
if user_input:
    # 사용자 메시지를 먼저 출력
    with st.chat_message("user"):
        st.write(user_input)
    
    # 세션 상태에 사용자 메시지 저장
    st.session_state.messages.append({
        'role': 'user', 
        'content': user_input
    })

    if "qa_chain" not in st.session_state:
        with st.chat_message("assistant"):
            st.write("❌ 문서 처리가 완료되지 않았습니다. PDF를 먼저 업로드하세요.")
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': "❌ 문서 처리가 완료되지 않았습니다. PDF를 먼저 업로드하세요."
        })
        st.stop()

    # 답변 생성 중 로딩 표시
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                response = st.session_state.qa_chain.invoke({"input": user_input})
                answer = response["answer"]
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}"
        
        st.write(answer)

    # 세션 상태에 어시스턴트 메시지 저장
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': answer
    })