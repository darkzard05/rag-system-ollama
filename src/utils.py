import os
import torch
import time
import subprocess
import logging
import functools
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import streamlit as st
from typing import List, Optional, Dict
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 모델 및 설정 상수
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 리트리버 설정 상수
RETRIEVER_CONFIG: Dict = {
    'search_type': "mmr",
    'search_kwargs': {
        'k': 5,           # 검색 결과 수 최적화
        'fetch_k': 20,    # 후보 수 증가
        'lambda_mult': 0.8 # MMR 다양성 가중치 증가
    }
}

# 텍스트 분할 설정
TEXT_SPLITTER_CONFIG: Dict = {
    'chunk_size': 4000,
    'chunk_overlap': 200,
}

class SessionManager:
    """세션 상태를 관리하는 클래스"""
    
    # 세션 상태의 기본값을 클래스 변수로 정의
    DEFAULT_SESSION_STATE = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "temp_pdf_path": None,
        "pdf_is_processing": False,
        "processing_step": None,
        "source_documents": {} # source_documents 초기화 추가
    }
    
    @classmethod
    def init_session(cls):
        """세션 상태 초기화 - 한 번만 실행되어야 함"""
        if not st.session_state.get("_initialized", False):
            logging.info("세션 상태 초기화 중...")
            for key, value in cls.DEFAULT_SESSION_STATE.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            st.session_state._initialized = True
    
    @classmethod
    def reset_session_state(cls, keys=None):
        """지정된 키들의 세션 상태를 기본값으로 리셋"""
        keys_to_reset = keys if keys is not None else cls.DEFAULT_SESSION_STATE.keys()
        for key in keys_to_reset:
            if key in cls.DEFAULT_SESSION_STATE:
                st.session_state[key] = cls.DEFAULT_SESSION_STATE[key]
    
    @classmethod
    def reset_for_new_file(cls, uploaded_file):
        """새 파일 업로드시 세션 상태 리셋"""
        logging.info("새 파일 업로드로 인한 세션 상태 리셋 중...")
        
        # 모델 변경 메시지 저장
        last_model_change_message = st.session_state.get("last_model_change_message")
        
        file_related_keys = [
            "last_uploaded_file_name",
            "pdf_processed",
            "pdf_processing_error",
            "qa_chain",
            "vector_store",
            "pdf_is_processing",
            "processing_step",
            "messages"
        ]
        cls.reset_session_state(file_related_keys)
        st.session_state.last_uploaded_file_name = uploaded_file.name
        
        # 모델 변경 메시지 복원
        if last_model_change_message:
            st.session_state.last_model_change_message = last_model_change_message
            cls.add_message("assistant", last_model_change_message)
        
        # Streamlit 캐시 초기화
        st.cache_data.clear()
        st.cache_resource.clear()
    
    @classmethod
    def add_message(cls, role: str, content: str):
        """메시지 추가"""
        if not st.session_state.get("messages"):
            st.session_state.messages = []
        st.session_state.messages.append({"role": role, "content": content})
    
    @classmethod
    def update_progress(cls, step: str, message: str):
        """처리 단계 업데이트 및 진행 상황 메시지 표시"""
        st.session_state.processing_step = step
        cls.add_message("assistant", f"🔄 {message}")
    
    @staticmethod
    def is_ready_for_chat():
        """채팅 준비 상태 확인"""
        return (st.session_state.get("pdf_processed") and 
                not st.session_state.get("pdf_processing_error") and 
                st.session_state.get("qa_chain") is not None)
    
    @classmethod
    def update_model(cls, new_model: str):
        """모델 업데이트"""
        old_model = st.session_state.get("last_selected_model", "N/A")
        model_related_keys = ["last_selected_model", "llm", "qa_chain"]
        cls.reset_session_state(model_related_keys)
        st.session_state.last_selected_model = new_model
        
        cls.add_message(
            "assistant", 
            f"🔄 모델을 {old_model}에서 {new_model}로 변경하는 중..."
        )
        return old_model

    @classmethod
    def handle_error(cls, error: Exception, error_context: str, affected_states: list = None):
        """에러 처리 및 상태 업데이트"""
        error_msg = f"{error_context}: {str(error)}"
        logging.error(error_msg, exc_info=True)
        
        if affected_states:
            cls.reset_session_state(affected_states)
            
        cls.add_message("assistant", f"❌ {error_msg}")
        return error_msg
    
    @classmethod
    def set_error_state(cls, error_message: str, error_context: str = None):
        """에러 상태 설정"""
        st.session_state.pdf_processing_error = error_message
        if error_context:
            logging.error(f"{error_context}: {error_message}")
        cls.add_message("assistant", f"❌ {error_message}")
    
    @classmethod
    def clear_error_state(cls):
        """에러 상태 초기화"""
        st.session_state.pdf_processing_error = None

# 로깅 데코레이터 수정
def log_operation(operation_name):
    def decorator(func):
        @functools.wraps(func)  # 함수 메타데이터 보존
        def wrapper(*args, **kwargs):
            logging.info(f"{operation_name} 시작...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                logging.info(f"{operation_name} 완료 (소요 시간: {time.time() - start_time:.2f}초)")
                return result
            except Exception as e:
                logging.error(f"{operation_name} 중 오류 발생: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

@st.cache_data(show_spinner=False)
@log_operation("Ollama 모델 목록 불러오기")
def get_ollama_models() -> List[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    return [line.split()[0] for line in result.stdout.split("\n")[1:] if line]

@st.cache_resource(show_spinner=False)
@log_operation("PDF 파일 로드")
def load_pdf_docs(pdf_file_path: str) -> List:
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
    loader = PyMuPDFLoader(
        pdf_file_path,
        mode="page",
        )
    return loader.load()

@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, # 정의된 상수 사용
        model_kwargs={
            "device": device,
            "trust_remote_code": False,
            },
        encode_kwargs={
            "device": device,
            "batch_size": 32,
            "normalize_embeddings": True,
            },
    )
    return embedder

@st.cache_data(show_spinner=False)
@log_operation("문서 분할")
def split_documents(_docs: List) -> List:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'],
        chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap'],
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        is_separator_regex=False,
        add_start_index=True,
    )
    return chunker.split_documents(_docs)

@st.cache_resource(show_spinner=False)
@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(_documents, _embedder) -> Optional[FAISS]:
    return FAISS.from_documents(
        documents=_documents,
        embedding=_embedder,
    )

@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_llm(model_name: str) -> OllamaLLM:
    return OllamaLLM(
        model=model_name,
        num_predict=-1,
        )

# QA 프롬프트를 함수 외부에서 정의하여 다른 모듈에서 import 가능하게 함
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
     """당신은 주어진 컨텍스트만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다. 다른 지식이나 정보를 사용해서는 안 됩니다.

     **컨텍스트:**
     {context}

     **답변 생성 지침:**
     1.  **언어:** 사용자의 질문과 동일한 언어로 답변해야 합니다.
     2.  **답변 형식 (`answer` 필드 내용):**
         *   답변은 반드시 마크다운 형식으로, 명확하게 작성해야 합니다.
         *   **출처 표기 위치:** 컨텍스트에서 가져온 정보에 대한 출처 번호(예: `[1]`, `[2]`)는, 해당 정보가 기술된 **구절이나 문장의 끝에** 명시해야 합니다. (예: "정보 A는 문서에서 설명하고 있습니다[1]. 또한 정보 B도 언급됩니다[2].")
             *   **줄 바꿈 및 가독성:**
                 *   답변 내용이 여러 항목, 단계 또는 문단으로 구성될 경우, 마크다운의 줄 바꿈(예: 빈 줄 삽입)이나 목록(숫자 목록, 글머리 기호 목록)을 적절히 사용하여 가독성을 높여야 합니다.
                 *   각 정보 단위가 명확히 구분되도록 표현해야 합니다.
                 *   **예시 (여러 항목을 설명하는 경우):**
                     ```markdown
                     문서의 주요 내용은 다음과 같습니다:
                     1.  **주제 A**: 주제 A에 대한 상세 설명입니다. 이 부분은 문서의 핵심 내용을 다룹니다[1].
                     2.  **주제 B**: 주제 B는 다른 관점을 제공하며, 관련된 예시를 포함합니다[2].
                         *   부연 설명: 주제 B의 특정 측면에 대한 추가 정보입니다.
                     3.  **결론**: 문서의 결론은 이러한 주제들을 종합하여 요약합니다[3].
                     ```
                 *   위 예시처럼, 최종적으로 렌더링될 때 깔끔하고 이해하기 쉬운 답변을 생성하는 것을 목표로 합니다.
             *   **매우 중요:** 출처 번호는 **절대로** 문장이나 구절의 시작 부분에 와서는 안 됩니다.
                 *   **잘못된 예시:** `[1] 이 정보는 중요합니다.`
                 *   **올바른 예시:** `이 정보는 중요합니다[1].`
         *   **다중 출처:** 하나의 정보에 여러 출처가 있다면 `[1][2]`와 같이 연달아 표시합니다.
         *   **번호 일치 및 일관성:** `answer` 필드 내의 출처 번호는 제공되는 컨텍스트의 `[번호]`와 일치해야 하며, `sources` 필드의 `id`와도 일치해야 합니다. (예: 컨텍스트가 `[1] 첫 번째 문서 내용... \n[2] 두 번째 문서 내용...`으로 제공되면, 답변에서 첫 번째 문서를 인용 시 `[1]`을 사용하고, `sources` 배열의 해당 객체 `id`는 `1`이어야 합니다.) `answer` 필드에서 사용된 모든 출처 번호는 `sources` 배열에 해당 `id`를 가진 객체로 반드시 포함되어야 하며, 그 반대로 `sources` 배열에 있는 모든 `id`는 `answer` 필드 어딘가에 `[id]` 형태로 인용되어야 합니다.

     3.  **정보 출처 (`sources` 필드 내용):**
         *   `answer` 필드에서 **실제로 인용한** 각 컨텍스트 문서에 대해 `sources` 배열에 객체를 추가해야 합니다.
         *   각 소스 객체는 다음 필드를 가져야 합니다:
             *   `id` (정수): `answer`에서 인용된 컨텍스트 문서의 번호입니다. (예: `[1]`을 인용했다면 `1`)
             *   `text` (문자열): `answer`에서 해당 `[id]`로 인용한 정보의 **직접적인 근거가 되는 컨텍스트 원문의 핵심 구절**이어야 합니다. 이 구절은 사용자가 인용의 타당성을 빠르게 확인할 수 있도록 도와야 하며, 너무 길거나 짧지 않게, **가장 관련성이 높은 부분**을 정확히 발췌해야 합니다.
             *   `page` (문자열): 해당 컨텍스트 문서의 페이지 번호입니다. 페이지 정보가 컨텍스트에 명시되어 있다면 그 값을 사용하고, 없다면 "N/A"로 표시합니다. (컨텍스트는 `(p.페이지번호)` 형식으로 페이지 정보를 포함할 수 있습니다.)
     4.  **답변 불가 시:**
         *   만약 질문에 대한 답변을 제공된 컨텍스트 내에서 찾을 수 없다면, `answer` 필드에 "제공된 컨텍스트에서는 질문에 대한 답변을 찾을 수 없습니다."라고 명확히 답변해야 합니다.
         *   이 경우, `sources` 배열은 빈 배열 `[]`이어야 합니다.

     **최종 JSON 출력 형식:**
     응답은 반드시 다음 JSON 형식이어야 합니다. **JSON 객체 자체만을 반환해야 하며, JSON 객체를 감싸는 마크다운(예: ```json ... ```)이나 다른 설명 텍스트를 포함해서는 안 됩니다.**
     ```json
     {{
       "answer": answer,
       "sources": [
         {{
           "id": id,
           "text": text,
           "page": page,
         }},
       ]
     }}
     ```
     만약 답변을 찾을 수 없는 경우의 예시:
     ```json
     {{
       "answer": "제공된 컨텍스트에서는 질문에 대한 답변을 찾을 수 없습니다.",
       "sources": []
     }}
     ```
     """
        )),
    ("human", "Question: {input}")
    ])


def update_qa_chain(llm, vector_store):
    """QA 체인 업데이트"""
    try:
        # 헬퍼 함수 정의
        def init_source_docs_and_pass_input(input_data: Dict) -> Dict:
            """세션 상태의 source_documents를 초기화하고 입력을 그대로 반환합니다."""
            st.session_state.source_documents = {}
            return input_data

        def add_doc_number_to_metadata_and_save(docs: List[Dict]) -> List[Dict]:
            """
            검색된 각 문서에 'doc_number' 메타데이터를 추가하고,
            st.session_state.source_documents에 저장합니다.
            """
            # Ensure source_documents exists and is a dictionary.
            # This is a safeguard, as init_source_docs_and_pass_input should handle this.
            if not isinstance(st.session_state.get("source_documents"), dict):
                st.session_state.source_documents = {}
            for i, doc in enumerate(docs, 1):
                doc.metadata["doc_number"] = i # 나중에 document_prompt에서 사용

                # 페이지 번호 처리: 0-indexed를 1-indexed 문자열로 변환 또는 'N/A'
                # PyMuPDFLoader는 'page' 메타데이터를 0-indexed 정수로 제공
                page_number = doc.metadata.get('page')
                if page_number is not None:
                    doc.metadata['page'] = str(page_number + 1) # 1-indexed 문자열로 덮어쓰기
                else:
                    doc.metadata['page'] = 'N/A'

                # UI 툴팁 및 참조를 위해 세션 상태에 저장
                st.session_state.source_documents[str(i)] = {
                    'content': doc.page_content.strip(),
                    'page': doc.metadata.get('page') # 이제 이 값은 1-indexed 문자열 또는 'N/A'
                }
            return docs

        def rename_documents_key(data_dict: Dict) -> Dict:
            """'processed_documents' 키를 'documents'로 변경합니다."""
            if "processed_documents" in data_dict:
                data_dict["documents"] = data_dict.pop("processed_documents")
            return data_dict

        # 리트리버 설정
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG['search_type'],
            search_kwargs=RETRIEVER_CONFIG['search_kwargs']
        )

        # 각 문서를 LLM 프롬프트의 컨텍스트 부분에 맞게 포맷팅하기 위한 프롬프트
        # add_doc_number_to_metadata_and_save 함수에서 doc.metadata에 'doc_number'와 'page'가 설정됨
        document_prompt = PromptTemplate.from_template(
            "[{doc_number}] {page_content} (p.{page})"
        )

        # LLM에 최종적으로 전달될 프롬프트를 사용하여 문서 결합 체인 생성
        # create_stuff_documents_chain은 QA_PROMPT의 {context}를 document_prompt로 포맷된 문서들로 채움
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=QA_PROMPT,
            document_prompt=document_prompt,
            document_separator="\n\n",
            document_variable_name="context" # Explicitly use "context" from QA_PROMPT
        )

        # LCEL을 사용하여 전체 RAG 체인 구성
        # 1. 세션 상태 초기화 -> 2. 입력 통과 및 문서 검색/처리 -> 3. 키 이름 변경 -> 4. LLM 호출
        # retriever가 문자열 입력을 받도록 RunnableLambda를 사용하여 'input' 키의 값을 추출
        retrieval_chain_with_processing = RunnablePassthrough.assign(
            processed_documents=RunnableLambda(lambda x: x["input"]) # 'input' 키의 값만 retriever로 전달
                                | retriever
                                | RunnableLambda(add_doc_number_to_metadata_and_save)
        )

        final_qa_chain = (
            RunnableLambda(init_source_docs_and_pass_input) # 입력: {"input": "question"}
            | retrieval_chain_with_processing # 출력: {"input": "question", "processed_documents": [docs_with_metadata]}
            # Ensure the key for documents matches what combine_docs_chain expects via QA_PROMPT's {context}
            | RunnableLambda(lambda x: {"input": x["input"], "context": x.pop("processed_documents")}) 
            # 이전 rename_documents_key 대신 context로 직접 매핑
            # | RunnableLambda(rename_documents_key) # 출력: {"input": "question", "documents": [docs_with_metadata]}
            | combine_docs_chain # 입력: {"input": "question", "documents": [docs]}, 출력: LLM 답변 문자열 (스트리밍 시 청크)
        )
        return final_qa_chain

    except Exception as e:
        raise ValueError(f"QA 체인 업데이트 실패: {e}")

def process_pdf(uploaded_file, selected_model: str, temp_pdf_path: str):
    """PDF 처리 및 QA 체인 생성."""
    try:
        # 상태 초기화
        st.session_state.pdf_is_processing = True
        st.session_state.pdf_processed = False
        st.session_state.qa_chain = None
        
        # 각 단계 처리
        docs = load_pdf_docs(temp_pdf_path)
        embedder = load_embedding_model()
        documents = split_documents(docs)
        vector_store = create_vector_store(documents, embedder)
        llm = load_llm(selected_model)
        
        # QA 체인 생성
        qa_chain = update_qa_chain(llm, vector_store)
        
        # 세션 상태 한번에 업데이트
        st.session_state.update({
            'vector_store': vector_store,
            'llm': llm,
            'qa_chain': qa_chain,
            'pdf_processed': True,
            'pdf_processing_error': None
        })
        
        # 성공 메시지
        success_message = (
            f"✅ '{uploaded_file.name}' 문서 처리가 완료되었습니다.\n\n"
            "다음과 같은 질문들을 해보세요:\n\n"
            "[문서 전체 이해하기]\n"
            "- 이 문서를 한 문단으로 요약해주세요\n"
            "- 이 문서의 주요 주장과 근거를 설명해주세요\n"
            "- 이 문서의 핵심 용어 3가지를 설명해주세요\n\n"
            "[세부 내용 파악하기]\n"
            "- 이 문서가 해결하고자 하는 문제는 무엇인가요?\n"
            "- 문서에서 제시된 해결책이나 제안은 무엇인가요?\n"
            "- 이 연구의 한계점이나 향후 연구 방향은 무엇인가요?\n\n"
            "자유롭게 문서의 내용에 대해 질문해보세요."
        )
        SessionManager.add_message("assistant", success_message)

    except Exception as e:
        SessionManager.handle_error(
            error=e,
            error_context="PDF 처리",
            affected_states=['pdf_processed', 'qa_chain', 'vector_store', 'llm']
        )
        raise
    finally:
        st.session_state.pdf_is_processing = False
        st.session_state.processing_step = None
    
    st.rerun()