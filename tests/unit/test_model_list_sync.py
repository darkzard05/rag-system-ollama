import pytest
from unittest.mock import patch
from main import _load_available_models, _is_ollama_reachable
from common.config import DEFAULT_OLLAMA_MODEL, MSG_ERROR_OLLAMA_NOT_RUNNING


def test_model_list_sync_and_reachable():
    """전역 캐시 업데이트 시 모델 목록 동기화 및 Ollama 도달성 판별 검증."""
    with patch(
        "main._get_available_models_cached",
        return_value=["qwen3:4b", "deepseek-r1:7b"],
    ):
        models = _load_available_models(non_blocking_fallback=False)
        assert "qwen3:4b" in models
        assert "deepseek-r1:7b" in models
        assert _is_ollama_reachable() is True

    with patch(
        "main._get_available_models_cached",
        return_value=[MSG_ERROR_OLLAMA_NOT_RUNNING],
    ):
        models = _load_available_models(non_blocking_fallback=False)
        assert models == [DEFAULT_OLLAMA_MODEL]
        assert _is_ollama_reachable() is False
