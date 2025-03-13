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

st.set_page_config(page_title="RAG Chatbot", layout="wide")  # 페이지 설정

st.title("📄 RAG Chatbot with Ollama LLM")

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 초기 출력 메시지를 위한 공간 예약
initial_output = st.empty()

# 사이드바에 파일 업로드 배치
with st.sidebar:
    st.header("파일 업로드")
    uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

# 메인 페이지에 채팅 기록 배치
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# PDF 로딩을 캐싱: 파일 바이트를 받아 문서를 추출
@st.cache_data(show_spinner=False)
def load_pdf_docs(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        os.remove(temp_path)  # 파일 삭제를 로드 후로 이동
        return docs
    except Exception as e:
        logging.error(f"PDF 로드 중 오류 발생: {e}")
        return []

# 문서 분할 캐싱: 로드된 문서를 청크 단위로 분할
@st.cache_data(show_spinner=False)
def split_documents(_docs):
    try:
        embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2",
                                         model_kwargs={'device': 'cpu'})
        chunker = SemanticChunker(embedder)
        return chunker.split_documents(_docs)
    except Exception as e:
        logging.error(f"문서 분할 중 오류 발생: {e}")
        return []

# 벡터 저장소 생성 캐싱: 분할된 문서 청크를 기반으로 FAISS 벡터 저장소 구축
@st.cache_resource(show_spinner=False)
def create_vector_store(_documents):
    try:
        embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2",
                                         model_kwargs={'device': 'cpu'})
        return FAISS.from_documents(_documents, embedder)
    except Exception as e:
        logging.error(f"벡터 저장소 생성 중 오류 발생: {e}")
        return None

# LLM 초기화 캐싱: LLM 객체는 한 번만 생성하도록 함
@st.cache_resource(show_spinner=False)
def init_llm():
    try:  # 변경된 부분: 에러 처리 추가
        return OllamaLLM(model="gemma3:4b", device='cuda')
    except Exception as e:
        logging.error(f"LLM 초기화 중 오류 발생: {e}")
        return None

# uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    docs = load_pdf_docs(file_bytes)
    initial_output.write(f"📄 로드된 문서(페이지) 수: {len(docs)}")
    if not docs:
        st.error("❌ PDF에서 문서를 추출하지 못했습니다.")
        st.stop()

    with ThreadPoolExecutor() as executor:
        future_split = executor.submit(split_documents, docs)
        documents = future_split.result()
    
    initial_output.write(f"📑 분할된 문서 청크 수: {len(documents)}")

    vector_store = create_vector_store(documents)
    if vector_store is None:
        st.error("❌ 벡터 저장소 생성에 실패했습니다.")
        st.stop()
    
    initial_output.write("🗄️ FAISS 벡터 저장소가 생성되었습니다.")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 100})
    initial_output.write("🔍 검색기가 생성되었습니다.")

    llm = init_llm()
    if llm is None:
        st.error("❌ LLM 초기화에 실패했습니다.")
        st.stop()
    
    initial_output.write("🤖 LLM이 초기화되었습니다.")
    
    # # 문서가 성공적으로 로드된 후에 질문을 요청
    # st.session_state.messages.append({"role": "assistant", "content": "안녕하세요! PDF 문서 관련 질문을 해주세요."})
    # for msg in st.session_state.messages:
    #     st.chat_message(msg["role"]).write(msg["content"])

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

    # 사용자 입력 처리
    user_input = st.chat_input("메시지를 입력하세요")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        try:
            response = qa_chain.invoke({"input": user_input})
            answer = response["answer"]
        except Exception as e:
            answer = f"오류가 발생했습니다: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)