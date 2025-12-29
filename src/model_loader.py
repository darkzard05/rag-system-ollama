"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
"""

import os
import logging
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
        import ollama
        ollama_response = ollama.list()
        models = sorted([model["model"] for model in ollama_response.get("models", [])])
        if models:
            logger.info(f"Ollama 모델 목록 확보: {models}")
            return models
        return [MSG_ERROR_OLLAMA_NOT_RUNNING]
    except Exception as e:
        logger.warning(
            f"Ollama 모델 목록 조회 실패. 서버 상태를 확인하세요. 오류: {e}"
        )
        return [MSG_ERROR_OLLAMA_NOT_RUNNING]


def get_available_models() -> List[str]:
    """
    Ollama 서버에서 사용 가능한 LLM 모델 목록을 가져옵니다.

    Streamlit 캐싱이 적용되며, 첫 호출 시만 서버에서 모델을 조회합니다.

    Returns:
        List[str]: 사용 가능한 Ollama 모델 이름 목록.
    """
    import streamlit as st

    @st.cache_data(ttl=3600)
    def _cached_models():
        """내부 캐싱 함수."""
        ollama_models = _fetch_ollama_models()

        if not ollama_models or ollama_models[0] == MSG_ERROR_OLLAMA_NOT_RUNNING:
            logger.error(
                "사용 가능한 LLM 모델을 찾을 수 없습니다. 기본 목록을 사용합니다."
            )
            return ollama_models

        return ollama_models

    return _cached_models()


def _get_dynamic_batch_size(device: str) -> int:
    """
    GPU VRAM에 따라 동적으로 배치 크기를 결정합니다. (한 번만 계산, 이후 캐시됨)

    Args:
        device (str): 'cuda' 또는 'cpu'

    Returns:
        int: 적절한 배치 크기
    """
    if hasattr(_get_dynamic_batch_size, "_cached_batch_size"):
        return _get_dynamic_batch_size._cached_batch_size

    if device != "cuda":
        logger.info("CPU 환경 감지 (기본 배치 크기: 64).")
        batch_size = 64
    else:
        import torch
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"감지된 GPU VRAM: {total_vram_gb:.2f}GB")

            if total_vram_gb > 16:
                batch_size = 256
            elif total_vram_gb > 8:
                batch_size = 128
            elif total_vram_gb > 4:
                batch_size = 64
            else:
                batch_size = 32

            logger.info(f"VRAM 기반 동적 배치 크기 설정: {batch_size}")
        except Exception as e:
            logger.warning(f"VRAM 확인 실패: {e}. 기본 배치 크기(64)를 사용합니다.")
            batch_size = 64

    _get_dynamic_batch_size._cached_batch_size = batch_size
    return batch_size


def load_embedding_model(embedding_model_name: str) -> "HuggingFaceEmbeddings":
    """
    Hugging Face 임베딩 모델을 로드합니다. (첫 로드 후 캐시됨)

    Args:
        embedding_model_name (str): 로드할 Hugging Face 모델 이름.

    Returns:
        HuggingFaceEmbeddings: 로드된 임베딩 모델 인스턴스.
    """
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    @log_operation("임베딩 모델 로드")
    def _load_embedding_model_cached(model_name: str) -> "HuggingFaceEmbeddings":
        """임베딩 모델을 로드하고 캐싱합니다."""
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"임베딩 모델 실행 장치: {device}")

        batch_size = 128
        if isinstance(EMBEDDING_BATCH_SIZE, int):
            batch_size = EMBEDDING_BATCH_SIZE
            logger.info(f"설정 파일의 배치 크기 사용: {batch_size}")
        elif EMBEDDING_BATCH_SIZE == "auto":
            batch_size = _get_dynamic_batch_size(device)
        else:
            logger.warning(
                f"잘못된 배치 크기 설정 ('{EMBEDDING_BATCH_SIZE}'). 기본값(128)을 사용합니다."
            )

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": batch_size},
            cache_folder=CACHE_DIR,
        )

    return _load_embedding_model_cached(embedding_model_name)


def load_reranker_model(model_name: str):
    """
    RERANKER용 CrossEncoder 모델을 로드합니다. (캐시됨)

    Args:
        model_name (str): Hugging Face CrossEncoder 모델 이름.

    Returns:
        CrossEncoder: 로드된 모델 인스턴스.
    """
    import streamlit as st
    from sentence_transformers import CrossEncoder
    import torch

    @st.cache_resource(show_spinner=False)
    @log_operation("Reranker 모델 로드")
    def _load_reranker_model_cached(name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Reranker 모델 로드 중: '{name}' ({device})...")
        return CrossEncoder(name, device=device)

    return _load_reranker_model_cached(model_name)


@log_operation("Ollama LLM 로드")
def load_llm(
    model_name: str,
    temperature: float = OLLAMA_TEMPERATURE,
    num_predict: int = OLLAMA_NUM_PREDICT,
    top_p: float = OLLAMA_TOP_P,
    num_ctx: int = OLLAMA_NUM_CTX,
) -> "OllamaLLM":
    """
    Ollama LLM 모델을 로드합니다.

    Args:
        model_name (str): 로드할 모델의 이름.
        temperature (float): 모델의 온도 설정. 기본값은 config의 OLLAMA_TEMPERATURE.
        num_predict (int): 예측 토큰 수. 기본값은 config의 OLLAMA_NUM_PREDICT.
        top_p (float): Top-p 샘플링. 기본값은 config의 OLLAMA_TOP_P.
        num_ctx (int): 컨텍스트 윈도우. 기본값은 config의 OLLAMA_NUM_CTX.

    Returns:
        OllamaLLM: 로드된 Ollama LLM 인스턴스.

    Raises:
        ValueError: Ollama 서버가 실행 중이지 않을 때.
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