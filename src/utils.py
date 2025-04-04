import os
import time
import tempfile
import subprocess
import logging
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import streamlit as st

from typing import List, Optional, Dict, Any

@st.cache_data(show_spinner=False)
def get_ollama_models() -> List[str]:
    """Ollama 모델 목록을 가져오는 함수"""
    logging.info("Ollama 모델 목록을 불러오는 중...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.split("\n")[1:] if line]
        return models
    except Exception as e:
        logging.error(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}")
        return []

@st.cache_data(show_spinner=False)
def load_pdf_docs(file_path) -> List:
    """PDF 파일을 로드하는 함수"""
    logging.info("PDF 파일 로드 중...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_path)
            temp_path = tmp_file.name
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        os.remove(temp_path)
        return docs
    except Exception as e:
        logging.error(f"PDF 로드 중 오류 발생: {e}")
        return []

@st.cache_resource(show_spinner=False)
def get_embedder(model_name, model_kwargs=None, encode_kwargs=None) -> HuggingFaceEmbeddings:
    """HuggingFaceEmbeddings 모델을 초기화하는 함수"""
    return HuggingFaceEmbeddings(model_name=model_name,
                                 model_kwargs=model_kwargs,
                                 encode_kwargs=encode_kwargs)

@st.cache_data(show_spinner=False)
def split_documents(_docs: List, _embedder) -> List:
    """문서를 분할하는 함수"""
    logging.info("문서 분할 시작...")
    start_time = time.time()
    try:
        chunker = SemanticChunker(_embedder,
                                  buffer_size=3,
                                  min_chunk_size=512)
        docs = chunker.split_documents(_docs)
        logging.info(f"문서 분할 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return docs
    except Exception as e:
        logging.error(f"문서 분할 중 오류 발생: {e}")
        return []

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents, _embedder) -> Optional[FAISS]:
    """문서에서 벡터 저장소를 생성하는 함수"""
    logging.info("벡터 저장소 생성 중...")
    start_time = time.time()
    try:
        vector_space = FAISS.from_documents(_documents, _embedder)
        logging.info(f"벡터 저장소 생성 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return vector_space
    except Exception as e:
        logging.error(f"벡터 저장소 생성 중 오류 발생: {e}")
        return None

@st.cache_resource(show_spinner=False)
def init_llm(model_name) -> Optional[OllamaLLM]:
    """LLM을 초기화하는 함수"""
    logging.info("LLM 초기화 중...")
    try:
        return OllamaLLM(model=model_name, device='cuda')
    except Exception as e:
        logging.error(f"LLM 초기화 중 오류 발생: {e}")
        return None
