import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"
import torch
import time
import subprocess
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st

from typing import List, Optional
from langchain_core.messages import AIMessage

def init_session_state():
    """세션 상태 초기화 함수"""
    logging.info("세션 상태 초기화 중...")
    defaults = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "temp_pdf_path": None # 임시 PDF 파일 경로 저장용
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
def reset_session_state(uploaded_file):
    """세션 상태를 초기화합니다."""
    st.session_state.last_uploaded_file_name = uploaded_file.name
    st.session_state.pdf_processed = False
    st.session_state.pdf_processing_error = None
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.session_state.messages = []  # 새 파일이므로 채팅 기록 초기화
    load_pdf_docs.clear()
    split_documents.clear()
    create_vector_store.clear()

def prepare_chat_history():
    """이전 대화 기록을 준비합니다."""
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history

@st.cache_data(show_spinner=False)
def get_ollama_models() -> List[str]:
    """Ollama 모델 목록을 가져오는 함수"""
    logging.info("Ollama 모델 목록을 불러오기 시작...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.split("\n")[1:] if line]
        return models
    except Exception as e:
        logging.error(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}")
        raise ValueError(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}") from e

@st.cache_data(show_spinner=False)
def load_pdf_docs(pdf_file_path: str) -> List:
    """PDF 파일을 로드하는 함수"""
    logging.info("PDF 파일 로드 시작...")
    if not os.path.exists(pdf_file_path):
        logging.error(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
        raise FileNotFoundError(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
    try:
        loader = PyMuPDFLoader(pdf_file_path)
        docs = loader.load()
        logging.info(f"PDF 파일 로드 완료: {len(docs)} 페이지")
        return docs
    except Exception as e:
        logging.error(f"PDF 로드 중 오류 발생: {e}")
        raise ValueError(f"PDF 로드 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """HuggingFace 임베딩 모델을 로드하고 캐싱합니다."""
    logging.info("임베딩 모델 로딩 시작...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
    try:
        embedder = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",        #"BAAI/bge-m3" # 필요시 이 부분을 설정 가능하게 변경 가능
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'device': device}
        )
        logging.info("임베딩 모델 로딩 완료.")
        return embedder
    except Exception as e:
        logging.error(f"임베딩 모델 로딩 중 오류 발생: {e}", exc_info=True)
        raise ValueError(f"임베딩 모델 로딩 실패: {e}") from e

@st.cache_data(show_spinner=False)
def split_documents(_docs: List, _embedder) -> List:
    """문서를 분할하는 함수"""
    logging.info("문서 분할 시작...")
    start_time = time.time()
    try:
        chunker = SemanticChunker(_embedder)
        docs = chunker.split_documents(_docs)
        logging.info(f"문서 {len(docs)} 페이지 분할 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return docs
    except Exception as e:
        logging.error(f"문서 분할 중 오류 발생: {e}")
        raise ValueError(f"문서 분할 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents, _embedder, persist_directory: str = "chroma_db_store") -> Optional[Chroma]:
    """문서에서 Chroma 벡터 저장소를 생성하는 함수"""
    logging.info("Chroma 벡터 저장소 생성 시작...")
    start_time = time.time()
    try:
        # 디렉토리가 없으면 생성
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        vector_space = Chroma.from_documents(
            documents=_documents,
            embedding=_embedder,
            persist_directory=persist_directory # 저장 경로 지정
        )
        logging.info(f"Chroma 벡터 저장소 생성 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return vector_space
    except Exception as e:
        logging.error(f"Chroma 벡터 저장소 생성 중 오류 발생: {e}")
        raise ValueError(f"Chroma 벡터 저장소 생성 중 오류 발생: {e}") from e
    
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    """선택된 Ollama LLM을 로드하고 캐싱합니다."""
    logging.info(f"Ollama LLM 로딩 시작: {model_name}")
    try:
        llm = OllamaLLM(model=model_name)
        logging.info(f"Ollama LLM 로딩 완료: {model_name}")
        return llm
    except Exception as e:
        logging.error(f"Ollama LLM ({model_name}) 로딩 중 오류 발생: {e}", exc_info=True)
        raise ValueError(f"Ollama LLM ({model_name}) 로딩 실패: {e}") from e
    
def process_pdf(uploaded_file, selected_model, temp_pdf_path: str):
    """PDF 처리 및 QA 체인 생성."""
    try:
        docs = load_pdf_docs(temp_pdf_path)
        if not docs: raise ValueError("PDF 문서 로딩 실패")

        embedder = load_embedding_model()
        if not embedder: raise ValueError("임베딩 모델 로딩 실패 (캐시)")

        documents = split_documents(docs, embedder)
        if not documents: raise ValueError("문서 분할 실패")

        vector_store = create_vector_store(documents, embedder)
        if not vector_store: raise ValueError("벡터 저장소 생성 실패")
        st.session_state.vector_store = vector_store

        if isinstance(selected_model, str):
            llm = load_llm(selected_model)
            if not llm: raise ValueError("LLM 초기화 실패")
            st.session_state.llm = llm
        else:
            raise ValueError("LLM 초기화를 위한 모델 미선택")

        logging.info("QA 체인 생성 시작...")
        QA_PROMPT = ChatPromptTemplate.from_messages([
            ("system", ("당신은 질문 답변 과제를 위한 보조 AI입니다.\n"
                        "주어진 검색된 컨텍스트 조각을 사용하여 질문에 답하세요.\n"
                        "답을 모르면 모른다고 솔직하게 말하세요.\n\n"
                        "<context>\n{context}\n</context>")),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
            ])
        combine_chain = create_stuff_documents_chain(st.session_state.llm, QA_PROMPT)
        qa_chain = create_retrieval_chain(
            st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 7},
                ),
            combine_chain
        )
        logging.info("QA 체인 생성 완료.")
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_processed = True
        logging.info("PDF 처리 완료.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"✅ PDF 파일 '{uploaded_file.name}'의 문서 처리가 완료되었습니다.\n\n"
                "이제 문서 내용에 대해 자유롭게 질문해보세요. 예를 들면 다음과 같습니다:\n\n"
                "**기본 정보 확인:**\n"
                "- 이 문서의 핵심 주제는 무엇인가요?\n"
                "- 이 문서가 작성된 목적이 무엇인가요?\n"
                "- 문서를 간략하게 요약해주세요.\n\n"
                "**세부 내용 질문:**\n"
                "- [알고 싶은 특정 용어/개념]에 대해 설명해주세요.\n"
                "- 문서에서 [찾고 싶은 특정 정보/데이터] 부분을 찾아주세요.\n\n"
                "**구조 및 요점 파악:**\n"
                "- 이 문서의 주요 섹션(장)은 어떻게 구성되어 있나요?\n"
                "- 결론이나 주요 시사점은 무엇인가요?\n"
            )
        })
        return qa_chain, documents, embedder, vector_store

    except Exception as e:
        logging.error(f"PDF 처리 중 오류 발생: {e}", exc_info=True)
        st.session_state.pdf_processing_error = str(e)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ PDF 처리 중 오류가 발생했습니다: {e}"
        })
        return None, None, None, None