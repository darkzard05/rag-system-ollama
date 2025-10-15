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
    from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    CACHE_DIR,
    OLLAMA_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
    OLLAMA_NUM_PREDICT,
    PREFERRED_GEMINI_MODELS,
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

def _fetch_gemini_models() -> List[str]:
    # import google.generativeai as genai
    # if not GEMINI_API_KEY:
    #     return []
    # try:
    #     genai.configure(api_key=GEMINI_API_KEY)
    #     available_models_from_api = [
    #         m.name.replace("models/", "")
    #         for m in genai.list_models()
    #         if "generateContent" in m.supported_generation_methods
    #     ]
        
    #     filtered_gemini_models = [
    #         model
    #         for model in PREFERRED_GEMINI_MODELS
    #         if model in available_models_from_api
    #     ]

    #     if filtered_gemini_models:
    #         logging.info(f"선별된 Gemini 모델을 찾았습니다: {filtered_gemini_models}")
    #         return filtered_gemini_models
        
    #     fallback_models = [
    #         m
    #         for m in available_models_from_api
    #         if any(k in m for k in ["1.5", "pro"])
    #     ][:5]
    #     logging.info(
    #         f"선호하는 Gemini 모델을 찾지 못해, 사용 가능한 모델 중 일부를 사용합니다: {fallback_models}"
    #     )
    #     return fallback_models
    # except Exception as e:
    #     logging.warning(f"Gemini 모델 목록을 가져오는 데 실패했습니다: {e}")
    return []

@st.cache_data(ttl=3600)
def get_available_models() -> List[str]:
    ollama_models = _fetch_ollama_models()

    if not ollama_models:
        logging.error("사용 가능한 LLM 모델을 찾을 수 없습니다. 기본 모델 목록을 사용합니다.")
        return [OLLAMA_MODEL_NAME]

    return ollama_models


@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model(embedding_model_name: str) -> "HuggingFaceEmbeddings":
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    if hasattr(torch, "classes"):
        torch.classes.__path__ = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 128},
        cache_folder=CACHE_DIR,
    )


@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_ollama_llm(_model_name: str) -> "OllamaLLM":
    from langchain_ollama import OllamaLLM

    # --- 💡 JSON 및 temperature 전역 설정을 제거하고, 순수한 LLM 객체를 반환 💡 ---
    return OllamaLLM(model=_model_name, num_predict=OLLAMA_NUM_PREDICT)


# @st.cache_resource(show_spinner=False)
# @log_operation("Gemini LLM 로딩")
# def load_gemini_llm(_model_name: str) -> "ChatGoogleGenerativeAI":
#     from langchain_google_genai import ChatGoogleGenerativeAI

#     if not GEMINI_API_KEY:
#         raise ValueError("config.py 파일에 Gemini API 키를 설정해야 합니다.")
#     return ChatGoogleGenerativeAI(model=_model_name, google_api_key=GEMINI_API_KEY)


def load_llm(model_name: str):
    # if "gemini" in model_name.lower():
    #     return load_gemini_llm(_model_name=model_name)
    # else:
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