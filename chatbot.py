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

st.title("📄 RAG Chatbot with Ollama LLM")

# PDF 로딩을 캐싱: 파일 바이트를 받아 문서를 추출
@st.cache_data(show_spinner=False)
def load_pdf_docs(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        temp_path = tmp_file.name
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    os.remove(temp_path)
    return docs

# 문서 분할 캐싱: 로드된 문서를 청크 단위로 분할
@st.cache_data(show_spinner=False)
def split_documents(_docs):
    embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    chunker = SemanticChunker(embedder)
    return chunker.split_documents(_docs)

# 벡터 저장소 생성 캐싱: 분할된 문서 청크를 기반으로 FAISS 벡터 저장소 구축
@st.cache_resource(show_spinner=False, hash_funcs={torch.classes.__class__: lambda obj: None})
def create_vector_store(_documents):
    embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    return FAISS.from_documents(_documents, embedder)

# LLM 초기화 캐싱: LLM 객체는 한 번만 생성하도록 함
@st.cache_resource(show_spinner=False)
def init_llm():
    return OllamaLLM(model="deepseek-r1:14b", device='cuda')

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    docs = load_pdf_docs(file_bytes)
    st.write(f"📄 로드된 문서(페이지) 수: {len(docs)}")
    if not docs:
        st.error("❌ PDF에서 문서를 추출하지 못했습니다.")
        st.stop()

    documents = split_documents(docs)
    st.write(f"📑 분할된 문서 청크 수: {len(documents)}")

    vector_store = create_vector_store(documents)
    st.write("🗄️ FAISS 벡터 저장소가 생성되었습니다.")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 100})
    st.write("🔍 검색기가 생성되었습니다.")

    llm = init_llm()
    st.write("🤖 LLM이 초기화되었습니다.")

    prompt = """
    당신은 문서 기반 전문 조력자입니다. 다음 규칙을 엄격히 준수하세요:

    [규칙]
    1. 반드시 제공된 컨텍스트({context})만 사용
    2. 간결하고 명료한 한국어로 답변
    3. 전문적이고 깔끔한 형식으로 답변
    4. 불확실한 내용은 언급 금지

    [실제 질문]
    질문: {input}
    답변:
    """
    QA_PROMPT = PromptTemplate.from_template(prompt)
    combine_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    qa_chain = create_retrieval_chain(retriever, combine_chain)

    # 대화 이력 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! PDF 문서 관련 질문을 해주세요."}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 사용자 입력 처리
    user_input = st.chat_input("메시지를 입력하세요")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        response = qa_chain.invoke({"input": user_input})
        answer = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
