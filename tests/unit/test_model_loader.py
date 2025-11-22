import pytest
from unittest.mock import MagicMock, patch
from model_loader import (
    _fetch_ollama_models,
    get_available_models,
    load_embedding_model,
    load_ollama_llm,
    is_embedding_model_cached,
)
from config import MSG_ERROR_OLLAMA_NOT_RUNNING


@patch("model_loader.ollama")
def test_fetch_ollama_models_success(mock_ollama):
    """Tests fetching Ollama models successfully."""
    # 1. Arrange
    mock_ollama.list.return_value = {
        "models": [{"model": "model-b"}, {"model": "model-a"}]
    }

    # 2. Act
    models = _fetch_ollama_models()

    # 3. Assert
    assert models == ["model-a", "model-b"]


@patch("model_loader.ollama")
def test_fetch_ollama_models_failure(mock_ollama):
    """Tests fetching Ollama models when the server is not running."""
    # 1. Arrange
    mock_ollama.list.side_effect = Exception("Connection error")

    # 2. Act
    models = _fetch_ollama_models()

    # 3. Assert
    assert models == [MSG_ERROR_OLLAMA_NOT_RUNNING]


@patch("model_loader._fetch_ollama_models")
def test_get_available_models_success(mock_fetch):
    """Tests get_available_models when models are found."""
    # 1. Arrange
    mock_fetch.return_value = ["model-a", "model-b"]

    # 2. Act
    models = get_available_models()

    # 3. Assert
    assert models == ["model-a", "model-b"]


@patch("model_loader._fetch_ollama_models")
def test_get_available_models_failure(mock_fetch):
    """Tests get_available_models when no models are found."""
    # 1. Arrange
    mock_fetch.return_value = [MSG_ERROR_OLLAMA_NOT_RUNNING]

    # 2. Act
    models = get_available_models()

    # 3. Assert
    assert models == [MSG_ERROR_OLLAMA_NOT_RUNNING]


@patch("model_loader.HuggingFaceEmbeddings")
@patch("model_loader.torch.cuda.is_available", return_value=False)
def test_load_embedding_model(mock_is_available, mock_hf_embeddings):
    """Tests loading the embedding model."""
    # 1. Arrange
    model_name = "test-model"

    # 2. Act
    load_embedding_model(model_name)

    # 3. Assert
    mock_hf_embeddings.assert_called_once()


@patch("model_loader.OllamaLLM")
def test_load_ollama_llm(mock_ollama_llm):
    """Tests loading the Ollama LLM."""
    # 1. Arrange
    model_name = "test-model"

    # 2. Act
    load_ollama_llm(model_name)

    # 3. Assert
    mock_ollama_llm.assert_called_once_with(model=model_name, num_predict=-1)


@patch("model_loader.os.path.exists")
def test_is_embedding_model_cached(mock_exists):
    """Tests the embedding model cache check."""
    # 1. Arrange
    model_name = "user/model-name"
    mock_exists.return_value = True

    # 2. Act
    is_cached = is_embedding_model_cached(model_name)

    # 3. Assert
    mock_exists.assert_called_once_with(".model_cache/models--user--model-name")
    assert is_cached is True
