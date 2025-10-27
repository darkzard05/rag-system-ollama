"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
"""

import os
import logging
import streamlit as st
import ollama
from typing import List, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM

from config import (
    CACHE_DIR,
    OLLAMA_MODEL_NAME,
    OLLAMA_NUM_PREDICT,
    EMBEDDING_BATCH_SIZE,
)
from utils import log_operation


def _fetch_ollama_models() -> List[str]:
    try:
        ollama_response = ollama.list()
        models = sorted([model["model"] for model in ollama_response.get("models", [])])
        if models:
            logging.info(f"Ollama에서 다음 모델을 찾았습니다: {models}")
        return models
    except Exception as e:
        logging.warning(
            f"Ollama 모델 목록을 가져오는 데 실패했습니다. Ollama 서버가 실행 중인지 확인하세요. 오류: {e}"
        )
        return []



@st.cache_data(ttl=3600)
def get_available_models() -> List[str]:
    ollama_models = _fetch_ollama_models()

    if not ollama_models:
        logging.error("사용 가능한 LLM 모델을 찾을 수 없습니다. 기본 모델 목록을 사용합니다.")
        return [OLLAMA_MODEL_NAME]

    return ollama_models


def _get_dynamic_batch_size(device: str) -> int:
    """GPU VRAM에 따라 동적으로 배치 크기를 결정합니다."""
    if device != "cuda":
        logging.info("CPU 환경이므로 기본 배치 크기(64)를 사용합니다.")
        return 64

    import torch
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"사용 가능한 GPU VRAM: {total_vram_gb:.2f}GB")

        if total_vram_gb > 16:
            batch_size = 256
        elif total_vram_gb > 8:
            batch_size = 128
        elif total_vram_gb > 4:
            batch_size = 64
        else:
            batch_size = 32
        
        logging.info(f"VRAM 기반 동적 배치 크기: {batch_size}")
        return batch_size
    except Exception as e:
        logging.warning(f"VRAM 확인 중 오류 발생: {e}. 기본 배치 크기(64)를 사용합니다.")
        return 64


@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model(embedding_model_name: str) -> "HuggingFaceEmbeddings":
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    # if hasattr(torch, "classes"):
    #     torch.classes.__path__ = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")

    batch_size = 128  # 기본값
    if isinstance(EMBEDDING_BATCH_SIZE, int):
        batch_size = EMBEDDING_BATCH_SIZE
        logging.info(f"config.yml에 설정된 배치 크기({batch_size})를 사용합니다.")
    elif EMBEDDING_BATCH_SIZE == "auto":
        batch_size = _get_dynamic_batch_size(device)
    else:
        logging.warning(f"잘못된 배치 크기 설정('{EMBEDDING_BATCH_SIZE}'). 기본값(128)을 사용합니다.")

    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": batch_size},
        cache_folder=CACHE_DIR,
    )


@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_ollama_llm(_model_name: str) -> "OllamaLLM":
    from langchain_ollama import OllamaLLM

    # --- 💡 JSON 및 temperature 전역 설정을 제거하고, 순수한 LLM 객체를 반환 💡 ---
    return OllamaLLM(model=_model_name, num_predict=OLLAMA_NUM_PREDICT)





def load_llm(model_name: str):
    return load_ollama_llm(_model_name=model_name)


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