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
from langchain.chains import create_retrieval_chain
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

with st.sidebar:
    st.header("설정")
    models = get_ollama_models()
    selected_model = st.selectbox("사용할 Ollama 모델 선택", models) if models else st.text("Ollama 모델을 불러올 수 없습니다.")
    uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")
    
# 세션 상태 초기화 (새 문서 업로드 시 기존 상태 초기화)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs" not in st.session_state:
    st.session_state.docs = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# 기존 채팅 메시지 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if selected_model:
    with st.chat_message("assistant"):
        st.write(f"🛠️ 모델 `{selected_model}`을 선택했습니다.")

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

@st.cache_data(show_spinner=False)
def split_documents(_docs):
    try:
        embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2",
                                         model_kwargs={'device': 'cuda'})
        chunker = SemanticChunker(embedder)
        return chunker.split_documents(_docs)
    except Exception as e:
        logging.error(f"문서 분할 중 오류 발생: {e}")
        return []

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents):
    try:
        embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2",
                                         model_kwargs={'device': 'cuda'})
        return FAISS.from_documents(_documents, embedder)
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

# ✅ 새 PDF 파일 업로드 시 처리
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    
    with st.chat_message("assistant"):
        st.write("📂 PDF 파일을 처리 중입니다...")

    # 문서 로딩, 분할, 벡터 저장소 생성
    with st.spinner("📄 문서를 로드하는 중..."):
        docs = load_pdf_docs(file_bytes)
    if not docs:
        st.error("❌ PDF에서 문서를 추출하지 못했습니다.")
        st.stop()

    with st.chat_message("assistant"):
        st.write(f"📄 로드된 문서(페이지) 수: {len(docs)}")

    with st.spinner("📑 문서를 분할하는 중..."):
        with ThreadPoolExecutor() as executor:
            future_split = executor.submit(split_documents, docs)
            documents = future_split.result()
    
    with st.chat_message("assistant"):
        st.write(f"📑 분할된 문서 청크 수: {len(documents)}")

    with st.spinner("🗄️ 벡터 저장소를 생성하는 중..."):
        vector_store = create_vector_store(documents)
    if vector_store is None:
        st.error("❌ 벡터 저장소 생성에 실패했습니다.")
        st.stop()

    retriever = vector_store.as_retriever(search_type="similarity",
                                          search_kwargs={"k": 100})

    with st.chat_message("assistant"):
        st.write("🗄️ FAISS 벡터 저장소가 생성되었습니다.")
        st.write("🔍 검색기가 생성되었습니다.")

    if isinstance(selected_model, str):
        llm = init_llm(selected_model)
    else:
        llm = None
    
    if llm is None:
        st.error("❌ LLM 초기화에 실패했습니다.")
        st.stop()
    
    with st.chat_message("assistant"):
        st.write("🤖 LLM이 초기화되었습니다.")

    prompt = """
    당신은 문서 기반 전문 조력자입니다. 다음 규칙을 엄격히 준수하세요:

    [규칙]
    1. 반드시 제공된 컨텍스트({context})만 사용하여 답변하세요.
    2. 간결하고 명료한 한국어로 답변하세요.
    3. 전문적이고 깔끔한 형식으로 답변하세요.
    4. 불확실한 내용은 언급하지 마세요.

    [실제 질문]
    질문: {input}
    답변:
    """
    QA_PROMPT = PromptTemplate.from_template(prompt)
    combine_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    qa_chain = create_retrieval_chain(retriever, combine_chain)

# ✅ 사용자 입력 및 답변
user_input = st.chat_input("메시지를 입력하세요")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        try:
            response = qa_chain.invoke({"input": user_input})
            answer = response["answer"]
        except Exception as e:
            answer = f"오류가 발생했습니다: {e}"
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
