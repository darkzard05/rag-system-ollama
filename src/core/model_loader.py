"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

import os
import logging
import functools
from typing import List, TYPE_CHECKING, Optional

from common.typing_utils import T

import torch
import streamlit as st

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM
    from sentence_transformers import CrossEncoder

from common.exceptions import LLMInferenceError, EmbeddingModelError
from common.config import (
    CACHE_DIR,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    OLLAMA_TOP_P,
    OLLAMA_TIMEOUT,
    OLLAMA_BASE_URL,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)
from common.utils import log_operation
from services.monitoring.performance_monitor import get_performance_monitor, OperationType

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

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

from services.optimization.batch_optimizer import get_optimal_batch_size

@st.cache_resource(show_spinner=False)
def load_embedding_model(embedding_model_name: Optional[str] = None) -> "HuggingFaceEmbeddings":
    with monitor.track_operation(OperationType.EMBEDDING_GENERATION, {"model": embedding_model_name or "default"}) as op:
        from langchain_huggingface import HuggingFaceEmbeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Backward-compat: allow omitting model name in tests / legacy code
        if not embedding_model_name:
            from common.config import AVAILABLE_EMBEDDING_MODELS
            if not AVAILABLE_EMBEDDING_MODELS:
                raise EmbeddingModelError(
                    model="(none)",
                    reason="no_embedding_models_configured",
                    details={"msg": "AVAILABLE_EMBEDDING_MODELS가 비어 있습니다. config.yml을 확인하세요."},
                )
            embedding_model_name = AVAILABLE_EMBEDDING_MODELS[0]

        # 배치 사이즈 최적화
        batch_size = get_optimal_batch_size(device=device, model_type="embedding")
        logger.info(f"임베딩 로드: {embedding_model_name} ({device}, batch_size={batch_size})")
        
        result = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": batch_size},
            cache_folder=CACHE_DIR,
        )
        return result

@st.cache_resource(show_spinner=False)
def load_reranker_model(model_name: str) -> Optional["CrossEncoder"]:
    """
    [최적화] 리랭커 모델 로드
    - VRAM 경합 방지를 위해 디바이스 최적화 적용
    """
    try:
        from sentence_transformers import CrossEncoder
        
        # [최적화] 리랭커는 상대적으로 가볍고 LLM과 VRAM을 공유하면 성능이 저하되므로,
        # GPU 메모리가 넉넉하지 않은 경우 CPU 사용을 고려
        is_gpu, total_mem = 0, 0
        try:
            is_gpu = torch.cuda.is_available()
            if is_gpu:
                total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        except: pass

        # [수정] 6GB GPU 환경에서도 리랭커 가속을 위해 기준을 4GB로 하향
        device = "cuda" if (is_gpu and total_mem > 4000) else "cpu"
        
        logger.info(f"Reranker 로드: {model_name} (Device: {device})")
        return CrossEncoder(model_name, device=device)
    except Exception as e:
        logger.error(f"Reranker 로드 실패: {e}")
        raise EmbeddingModelError(
            model=model_name,
            reason="Reranker 모델 로드 실패",
            details={"error": str(e)}
        )

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
) -> "ChatOllama":
    with monitor.track_operation(OperationType.PDF_LOADING, {"model": model_name, "timeout": timeout}) as op:
        if not model_name or model_name == MSG_ERROR_OLLAMA_NOT_RUNNING:
            raise LLMInferenceError(
                model=model_name,
                reason="model_not_selected",
                details={"msg": "Ollama 모델이 선택되지 않았습니다."}
            )

        from core.custom_ollama import DeepThinkingChatOllama

        logger.debug(f"Ollama 로드 설정: predict={num_predict}, ctx={num_ctx}, temp={temperature}")

        # ChatOllama 사용으로 사고 과정(thinking) 필드 지원 강화
        # [Custom] DeepThinkingChatOllama를 사용하여 Ollama의 thinking 필드 캡처 지원
        result = DeepThinkingChatOllama(
            model=model_name,
            num_predict=num_predict,
            top_p=top_p,
            num_ctx=num_ctx,
            temperature=temperature,
            timeout=timeout,
            keep_alive="60m",
            base_url=OLLAMA_BASE_URL,
            streaming=True,
        )
        return result

def is_embedding_model_cached(model_name: str) -> bool:
    model_path_name = f"models--{model_name.replace('/', '--')}"
    cache_path = os.path.join(CACHE_DIR, model_path_name)
    return os.path.exists(cache_path)