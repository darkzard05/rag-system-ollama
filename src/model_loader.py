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
    from langchain_google_genai import ChatGoogleGenerativeAI

# --- 설정 파일에서 상수 임포트 ---
from config import (
    CACHE_DIR,
    OLLAMA_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
    OLLAMA_NUM_PREDICT,
    PREFERRED_GEMINI_MODELS,
)
from utils import log_operation


# --- 모델 목록 조회 ---
@st.cache_data(ttl=3600)
def get_available_models() -> List[str]:
    """Ollama와 Gemini에서 사용 가능한 모델 목록을 동적으로 가져와 정렬된 리스트로 반환합니다."""
    import google.generativeai as genai

    ollama_models = []
    gemini_models = []

    # 1. Ollama 로컬 모델 가져오기
    try:
        ollama_response = ollama.list()
        ollama_models = sorted(
            [model["model"] for model in ollama_response.get("models", [])]
        )
        if ollama_models:
            logging.info(f"Ollama에서 다음 모델을 찾았습니다: {ollama_models}")
    except Exception as e:
        logging.warning(
            f"Ollama 모델 목록을 가져오는 데 실패했습니다. Ollama 서버가 실행 중인지 확인하세요. 오류: {e}"
        )

    # 2. Gemini 모델 가져오기 (선별된 최신 모델)
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            available_models_from_api = [
                m.name.replace("models/", "")
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]
            filtered_gemini_models = [
                model
                for model in PREFERRED_GEMINI_MODELS
                if model in available_models_from_api
            ]

            if filtered_gemini_models:
                gemini_models = filtered_gemini_models
                logging.info(f"선별된 Gemini 모델을 찾았습니다: {gemini_models}")
            else:
                fallback_models = [
                    m
                    for m in available_models_from_api
                    if any(k in m for k in ["1.5", "pro"])
                ][:5]
                gemini_models = fallback_models
                logging.info(
                    f"선호하는 Gemini 모델을 찾지 못해, 사용 가능한 모델 중 일부를 사용합니다: {fallback_models}"
                )
        except Exception as e:
            logging.warning(f"Gemini 모델 목록을 가져오는 데 실패했습니다: {e}")

    # 3. 최종 모델 목록 조합
    final_models = []
    if ollama_models:
        final_models.extend(ollama_models)

    if ollama_models and gemini_models:
        final_models.append("--------------------")  # 구분선 추가

    if gemini_models:
        final_models.extend(gemini_models)

    # 모델을 전혀 찾지 못한 경우 기본값 사용
    if not final_models:
        logging.error(
            "사용 가능한 LLM 모델을 찾을 수 없습니다. 기본 모델 목록을 사용합니다."
        )
        return [OLLAMA_MODEL_NAME, GEMINI_MODEL_NAME]

    return final_models


# --- 모델 로딩 ---
@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model(embedding_model_name: str) -> "HuggingFaceEmbeddings":
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    # Streamlit의 파일 감시 기능과 PyTorch 간의 호환성 문제를 해결하기 위한 임시 조치
    # Streamlit의 파일 감시 기능과 PyTorch 간의 호환성 문제를 해결하기 위한 임시 조치
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

    return OllamaLLM(model=_model_name, num_predict=OLLAMA_NUM_PREDICT)


@st.cache_resource(show_spinner=False)
@log_operation("Gemini LLM 로딩")
def load_gemini_llm(_model_name: str) -> "ChatGoogleGenerativeAI":
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GEMINI_API_KEY:
        raise ValueError("config.py 파일에 Gemini API 키를 설정해야 합니다.")
    return ChatGoogleGenerativeAI(model=_model_name, google_api_key=GEMINI_API_KEY)


def load_llm(model_name: str):
    """선택된 모델 이름에 따라 적절한 LLM을 로드합니다."""
    if "gemini" in model_name.lower():
        return load_gemini_llm(_model_name=model_name)
    else:
        return load_ollama_llm(_model_name=model_name)


def is_embedding_model_cached(model_name: str) -> bool:
    """지정된 임베딩 모델이 로컬 캐시에 존재하는지 확인합니다."""
    # Hugging Face의 캐시 경로 규칙을 따릅니다.
    # 예: "sentence-transformers/all-MiniLM-L6-v2" -> ".model_cache/models--sentence-transformers--all-MiniLM-L6-v2"
    model_path_name = f"models--{model_name.replace('/', '--')}"
    cache_path = os.path.join(CACHE_DIR, model_path_name)
    return os.path.exists(cache_path)
