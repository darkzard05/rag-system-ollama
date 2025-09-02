"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

import os
import logging
import hashlib
import json
from typing import List, Optional, Dict, Tuple
import tempfile

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# --- 설정 파일에서 상수 임포트 ---
from config import (
    RETRIEVER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
    QA_SYSTEM_PROMPT,
)
from session import SessionManager
from utils import log_operation


# --- 문서 처리 ---
@log_operation("PDF 문서 로드")
def load_pdf_docs(pdf_file_bytes: bytes) -> List:
    """PDF 파일 바이트에서 문서를 로드합니다."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        temp_file.write(pdf_file_bytes)
        temp_file.seek(0)
        loader = PyMuPDFLoader(file_path=temp_file.name)
        docs = loader.load()

    return docs


@log_operation("문서 분할")
def split_documents(docs: List) -> List:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
    )
    return chunker.split_documents(docs)


# --- 벡터 저장소 캐싱 ---
class VectorStoreCache:
    """벡터 저장소 캐싱을 관리하는 클래스"""

    def __init__(self, file_bytes: bytes, embedding_model_name: str):
        self.cache_dir, self.doc_splits_path = self._get_cache_paths(
            file_bytes, embedding_model_name
        )

    def _get_cache_paths(
        self, file_bytes: bytes, embedding_model_name: str
    ) -> Tuple[str, str]:
        """파일 내용과 임베딩 모델 이름 기반으로 고유 캐시 경로 생성"""
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        model_name_slug = embedding_model_name.replace("/", "_")
        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}"
        )
        doc_splits_path = os.path.join(cache_dir, "doc_splits.json")
        return cache_dir, doc_splits_path

    def _serialize_docs(self, docs: List[Document]) -> List[Dict]:
        """Document 객체 리스트를 JSON 직렬화 가능한 딕셔너리 리스트로 변환"""
        return [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]

    def _deserialize_docs(self, docs_as_dicts: List[Dict]) -> List[Document]:
        """딕셔너리 리스트를 Document 객체 리스트로 변환"""
        return [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in docs_as_dicts
        ]

    @log_operation("벡터 저장소 캐시 로드")
    def load(self, embedder) -> Optional[Tuple[List, FAISS]]:
        """디스크에서 FAISS 인덱스와 문서 조각을 로드"""
        if os.path.exists(self.cache_dir) and os.path.exists(self.doc_splits_path):
            try:
                vector_store = FAISS.load_local(
                    self.cache_dir, embedder, allow_dangerous_deserialization=True
                )
                with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                    doc_splits_as_dicts = json.load(f)
                doc_splits = self._deserialize_docs(doc_splits_as_dicts)
                logging.info(f"캐시를 '{self.cache_dir}'에서 불러왔습니다.")
                return doc_splits, vector_store
            except Exception as e:
                logging.warning(f"캐시 로드 중 오류 발생: {e}. 캐시를 재생성합니다.")
                return None
        return None

    @log_operation("벡터 저장소 캐시 저장")
    def save(self, doc_splits: List, vector_store: FAISS):
        """FAISS 인덱스와 문서 조각을 디스크에 저장"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            vector_store.save_local(self.cache_dir)
            with open(self.doc_splits_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._serialize_docs(doc_splits), f, ensure_ascii=False, indent=4
                )
            logging.info(f"캐시를 '{self.cache_dir}'에 저장했습니다.")
        except Exception as e:
            logging.error(f"캐시 저장 중 오류 발생: {e}")


# --- 리트리버 및 벡터 저장소 생성 ---
@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(docs: List, embedder: "HuggingFaceEmbeddings") -> "FAISS":
    return FAISS.from_documents(docs, embedder)


@log_operation("BM25 리트리버 생성")
def create_bm25_retriever(docs: List, k: int) -> "BM25Retriever":
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


@log_operation("Ensemble 리트리버 생성")
def create_ensemble_retriever(faiss_retriever, bm25_retriever, weights: List[float]):
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=weights
    )


@log_operation("QA 체인 생성/업데이트")
def create_qa_chain(llm, vector_store, doc_splits: Optional[List] = None):
    """
    LLM의 원본 텍스트 출력을 그대로 반환하는 RAG QA 체인을 생성합니다.
    """
    # 1. 리트리버 설정
    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )

    if doc_splits:
        try:
            bm25_retriever = create_bm25_retriever(
                docs=doc_splits, k=RETRIEVER_CONFIG["search_kwargs"]["k"]
            )
            final_retriever = create_ensemble_retriever(
                faiss_retriever, bm25_retriever, RETRIEVER_CONFIG["ensemble_weights"]
            )
            logging.info("EnsembleRetriever (BM25 + FAISS) 생성 및 사용.")
        except Exception as e:
            final_retriever = faiss_retriever
            logging.warning(
                f"EnsembleRetriever 생성 실패: {e}. FAISS 리트리버만 사용합니다."
            )
    else:
        final_retriever = faiss_retriever
        logging.info("분할된 문서가 없어 FAISS 리트리버만 사용합니다.")

    # 2. 문서 포맷팅 함수
    def format_docs(docs):
        """검색된 문서 리스트를 하나의 문자열로 합칩니다."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc.metadata["doc_number"] = i + 1
            page_number = doc.metadata.get("page", "N/A")
            if page_number != "N/A":
                doc.metadata["page"] = str(int(page_number) + 1)

            formatted_docs.append(
                f"[{doc.metadata['doc_number']}] {doc.page_content} (p.{doc.metadata.get('page', 'N/A')})"
            )
        return "\n\n".join(formatted_docs)

    # 3. LLM에 전달될 최종 프롬프트 구성
    # System Prompt: config.yml에서 가져온 LLM의 역할 및 지시사항
    # Human Prompt: 사용자 질문과 검색된 문맥(context)을 포함하는 실제 요청
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            (
                "human",
                "Based on the context below, answer the following question.\n\nQuestion: {input}\n\n[Context]\n{context}",
            ),
        ]
    )

    # 4. 체인 구성
    retrieval_chain = (
        RunnableLambda(lambda x: x["input"]) 
        | final_retriever
        | RunnableLambda(format_docs)
    )

    final_qa_chain = (
        RunnablePassthrough.assign(context=retrieval_chain)
        | qa_prompt
        | llm
        | StrOutputParser()  # 출력을 문자열로 파싱
    )

    return final_qa_chain


# --- 전체 PDF 처리 파이프라인 ---
def process_pdf_and_build_chain(
    file_bytes: bytes, llm, embedder
) -> Tuple[List, FAISS, bool]:
    """PDF 처리부터 벡터 저장소 생성까지의 과정을 관리하고, 결과를 반환합니다."""
    cache = VectorStoreCache(file_bytes, embedder.model_name)
    cached_data = cache.load(embedder)

    if cached_data:
        doc_splits, vector_store = cached_data
        cache_used = True
    else:
        cache_used = False
        docs = load_pdf_docs(file_bytes)
        doc_splits = split_documents(docs)
        vector_store = create_vector_store(doc_splits, embedder)
        cache.save(doc_splits, vector_store)

    return doc_splits, vector_store, cache_used


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str, file_bytes: bytes, llm, embedder
) -> Tuple[str, bool]:
    """세션을 업데이트하고 RAG 파이프라인 전체를 구축합니다."""
    doc_splits, vector_store, cache_used = process_pdf_and_build_chain(
        file_bytes, llm, embedder
    )

    qa_chain = create_qa_chain(llm, vector_store, doc_splits)

    # 세션 업데이트
    SessionManager.set("processed_document_splits", doc_splits)
    SessionManager.set("vector_store", vector_store)
    SessionManager.set("qa_chain", qa_chain)
    SessionManager.set("pdf_processed", True)

    logging.info(
        f"'{uploaded_file_name}' 문서 처리 및 QA 체인 생성 완료. (캐시 사용: {cache_used})"
    )

    if cache_used:
        success_message = f"✅ '{uploaded_file_name}' 문서의 저장된 캐시를 불러왔습니다."
    else:
        success_message = (
            f"✅ '{uploaded_file_name}' 문서 처리가 완료되었습니다.\n\n"
            "이제 문서 내용에 대해 자유롭게 질문해보세요."
        )
    return success_message, cache_used


@log_operation("파이프라인의 LLM 업데이트")
def update_llm_in_pipeline(llm):
    """기존 RAG 파이프라인에서 LLM만 교체합니다."""
    vector_store = SessionManager.get("vector_store")
    doc_splits = SessionManager.get("processed_document_splits")

    if not all([vector_store, doc_splits]):
        raise ValueError("RAG 파이프라인이 완전히 구축되지 않아 LLM을 업데이트할 수 없습니다.")

    qa_chain = create_qa_chain(llm, vector_store, doc_splits)
    SessionManager.set("llm", llm)
    SessionManager.set("qa_chain", qa_chain)
    logging.info(f"QA 체인이 새로운 LLM '{llm.model}'(으)로 업데이트되었습니다.")
