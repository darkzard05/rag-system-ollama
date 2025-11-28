"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
"""

import os
import logging
import streamlit as st
import ollama
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM

from config import (
    CACHE_DIR,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    OLLAMA_TOP_P,
    EMBEDDING_BATCH_SIZE,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)
from utils import log_operation


logger = logging.getLogger(__name__)


def _fetch_ollama_models() -> List[str]:
    """
    Ollama 서버에서 사용 가능한 LLM 모델 목록을 가져옵니다.
    
    Returns:
        List[str]: 사용 가능한 Ollama 모델 이름 목록.
    """
    try:
        ollama_response = ollama.list()
        models = sorted([model["model"] for model in ollama_response.get("models", [])])
        if models:
            logger.info(f"Found Ollama models: {models}")
            return models
        return [MSG_ERROR_OLLAMA_NOT_RUNNING]
    except Exception as e:
        logger.warning(
            f"Failed to fetch Ollama models. Is the Ollama server running? Error: {e}"
        )
        return [MSG_ERROR_OLLAMA_NOT_RUNNING]


@st.cache_data(ttl=3600)
def get_available_models() -> List[str]:
    """
    Ollama 서버에서 사용 가능한 LLM 모델 목록을 가져옵니다.
    
    Returns:
        List[str]: 사용 가능한 Ollama 모델 이름 목록.
    """
    ollama_models = _fetch_ollama_models()

    if not ollama_models or ollama_models[0] == MSG_ERROR_OLLAMA_NOT_RUNNING:
        logger.error(
            "Could not find any available LLM models. Using default list."
        )
        return ollama_models

    return ollama_models


def _get_dynamic_batch_size(device: str) -> int:
    """
    GPU VRAM에 따라 동적으로 배치 크기를 결정합니다.
    
    Args:
        device (str): 'cuda' 또는 'cpu'
    
    Returns:
        int: 적절한 배치 크기
    """
    if device != "cuda":
        logger.info("Running on CPU, using default batch size (64).")
        return 64

    import torch

    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Available GPU VRAM: {total_vram_gb:.2f}GB")

        if total_vram_gb > 16:
            batch_size = 256
        elif total_vram_gb > 8:
            batch_size = 128
        elif total_vram_gb > 4:
            batch_size = 64
        else:
            batch_size = 32

        logger.info(f"Using dynamic batch size based on VRAM: {batch_size}")
        return batch_size
    except Exception as e:
        logger.warning(
            f"Error checking VRAM: {e}. Using default batch size (64)."
        )
        return 64


@st.cache_resource(show_spinner=False)
@log_operation("Load embedding model")
def load_embedding_model(embedding_model_name: str) -> "HuggingFaceEmbeddings":
    """
    Hugging Face 임베딩 모델을 로드합니다.
    
    Args:
        embedding_model_name (str): 로드할 Hugging Face 모델 이름.
    
    Returns:
        HuggingFaceEmbeddings: 로드된 임베딩 모델 인스턴스.
    """
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    # if hasattr(torch, "classes"):
    #     torch.classes.__path__ = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for embedding model: {device}")

    batch_size = 128  # 기본값
    if isinstance(EMBEDDING_BATCH_SIZE, int):
        batch_size = EMBEDDING_BATCH_SIZE
        logger.info(f"Using batch size from config.yml: {batch_size}")
    elif EMBEDDING_BATCH_SIZE == "auto":
        batch_size = _get_dynamic_batch_size(device)
    else:
        logger.warning(
            f"Invalid batch size setting ('{EMBEDDING_BATCH_SIZE}'). Using default (128)."
        )

    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": batch_size},
        cache_folder=CACHE_DIR,
    )

@log_operation("Load Ollama LLM")
def load_llm(
    model_name: str,
    temperature: float = OLLAMA_TEMPERATURE,
    num_predict: int = OLLAMA_NUM_PREDICT,
    top_p: float = OLLAMA_TOP_P,
    num_ctx: int = OLLAMA_NUM_CTX,
    ) -> "OllamaLLM":
    """
    Ollama LLM 모델을 로드합니다.
    모든 설정값은 인자로 전달받으며, 전달되지 않을 경우 config.py의 상수를 기본값으로 사용합니다.
    """
    if model_name == MSG_ERROR_OLLAMA_NOT_RUNNING:
        raise ValueError("Ollama 서버가 실행 중이지 않아 모델을 로드할 수 없습니다.")
    
    from langchain_ollama import OllamaLLM
    
    return OllamaLLM(
        model=model_name, 
        num_predict=num_predict,
        top_p=top_p,
        num_ctx=num_ctx,
        temperature=temperature,
    )

def is_embedding_model_cached(model_name: str) -> bool:
    """
    주어진 Hugging Face 임베딩 모델이 로컬 캐시 디렉토리에 존재하는지 확인합니다.

    Args:
        model_name (str): 확인할 Hugging Face 모델의 이름 (예: "jhgan/ko-sroberta-multitask").

    Returns:
        bool: 모델이 캐시되어 있으면 True, 그렇지 않으면 False.
    """
    model_path_name = f"models--{model_name.replace('/', '--')}"
    cache_path = os.path.join(CACHE_DIR, model_path_name)
    return os.path.exists(cache_path)
