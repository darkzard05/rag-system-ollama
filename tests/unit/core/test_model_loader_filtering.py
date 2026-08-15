# ModelManager의 모델 필터링 로직을 테스트하는 유�� 테스트
import pytest
from core.model_loader import ModelManager
from common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL


def test_get_filtered_models_basic():
    """기본적인 모델 필터링 및 분류 테스트"""
    available_models = [
        "llama3:8b",
        "nomic-embed-text:latest",
        "mistral:latest",
        "bge-m3:latest",
        "---",
        "unknown-model",
    ]

    result = ModelManager.get_filtered_models(available_models)

    # LLM 모델 검증
    assert "llama3:8b" in result["llm"]
    assert "mistral:latest" in result["llm"]
    assert "unknown-model" in result["llm"]
    assert DEFAULT_OLLAMA_MODEL in result["llm"]

    # 임베딩 모델 검증
    assert "nomic-embed-text:latest" in result["embedding"]
    assert "bge-m3:latest" in result["embedding"]
    assert DEFAULT_EMBEDDING_MODEL in result["embedding"]

    # '---' 필터링 검증
    assert "---" not in result["llm"]
    assert "---" not in result["embedding"]


def test_get_filtered_models_keywords():
    """키워드 기반 임베딩 모델 분류 테스트"""
    available_models = [
        "my-embed-model",
        "bge-large",
        "nomic-v1",
        "mxbai-embed",
        "snowflake-arctic",
        "just-llm",
    ]

    result = ModelManager.get_filtered_models(available_models)

    # 임베딩으로 분류되어야 하는 모델들
    embed_expected = [
        "my-embed-model",
        "bge-large",
        "nomic-v1",
        "mxbai-embed",
        "snowflake-arctic",
    ]
    for model in embed_expected:
        assert model in result["embedding"]
        assert model not in result["llm"]

    assert "just-llm" in result["llm"]


def test_get_filtered_models_empty():
    """빈 모델 리스트 처리 테스트"""
    result = ModelManager.get_filtered_models([])
    assert result["llm"] == [DEFAULT_OLLAMA_MODEL]
    assert DEFAULT_EMBEDDING_MODEL in result["embedding"]


def test_get_filtered_models_duplicate_filtering():
    """중복 모델 필터링 테스트"""
    available_models = [
        "llama3:8b",
        "llama3:8b",
        "nomic-embed-text",
        "nomic-embed-text",
    ]

    result = ModelManager.get_filtered_models(available_models)

    # 중복 제거 확인
    assert result["llm"].count("llama3:8b") == 1
    assert result["embedding"].count("nomic-embed-text") == 1
