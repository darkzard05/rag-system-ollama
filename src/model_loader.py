"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

import os
import logging
import functools
from typing import List, TYPE_CHECKING, Optional

import torch
import streamlit as st

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM
    from sentence_transformers import CrossEncoder

from config import (
    CACHE_DIR,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    OLLAMA_TOP_P,
    OLLAMA_TIMEOUT,
    EMBEDDING_BATCH_SIZE,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)
from utils import log_operation

logger = logging.getLogger(__name__)

@st.cache_data(ttl=600) # 모델 목록 캐시 10분
def _fetch_available_models_cached() -> List[str]:
    try:
        import ollama
        ollama_response = ollama.list()
        models = []
        if hasattr(ollama_response, "models"):
            for model in ollama_response.models:
                name = getattr(model, "model", None) or (model.get("model") if isinstance(model, dict) else None)
                if name: models.append(name)
        elif isinstance(ollama_response, dict) and "models" in ollama_response:
             for model in ollama_response["models"]:
                 name = model.get("model") or model.get("name")
                 if name: models.append(name)
        models.sort()
        return models
    except Exception as e:
        logger.warning(f"Ollama 모델 목록 조회 실패: {e}")
        return []

@st.cache_resource(show_spinner=False)
def load_embedding_model(embedding_model_name: str) -> "HuggingFaceEmbeddings":
    from langchain_huggingface import HuggingFaceEmbeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"임베딩 로드: {embedding_model_name} ({device})")
    
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 64},
        cache_folder=CACHE_DIR,
    )

@st.cache_resource(show_spinner=False)
def load_reranker_model(model_name: str) -> Optional["CrossEncoder"]:
    try:
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return CrossEncoder(model_name, device=device)
    except Exception as e:
        logger.error(f"Reranker 로드 실패: {e}")
        return None

def get_available_models() -> List[str]:
    models = _fetch_available_models_cached()
    return models if models else [MSG_ERROR_OLLAMA_NOT_RUNNING]

@log_operation("Ollama LLM 로드")
def load_llm(
    model_name: str,
    temperature: float = OLLAMA_TEMPERATURE,
    num_predict: int = OLLAMA_NUM_PREDICT,
    top_p: float = OLLAMA_TOP_P,
    num_ctx: int = OLLAMA_NUM_CTX,
    timeout: float = OLLAMA_TIMEOUT,
) -> "OllamaLLM":
    if not model_name or model_name == MSG_ERROR_OLLAMA_NOT_RUNNING:
        raise ValueError("Ollama 모델이 선택되지 않았습니다.")

    from langchain_ollama import OllamaLLM

    logger.debug(f"Ollama 로드 설정: predict={num_predict}, ctx={num_ctx}, temp={temperature}")

    # 로컬 환경 최적화: 타임아웃 15분 설정, keep_alive 60분 설정
    return OllamaLLM(
        model=model_name,
        num_predict=num_predict,
        top_p=top_p,
        num_ctx=num_ctx,
        temperature=temperature,
        timeout=timeout,
        keep_alive="60m",  # 60분 동안 모델 메모리 유지
        base_url="http://127.0.0.1:11434",
        streaming=True,
    )

def is_embedding_model_cached(model_name: str) -> bool:
    model_path_name = f"models--{model_name.replace('/', '--')}"
    cache_path = os.path.join(CACHE_DIR, model_path_name)
    return os.path.exists(cache_path)