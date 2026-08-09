"""
TDD tests for inference timeout hardening (review R1b-01).

1. Semaphore acquisition must time out instead of blocking forever when the
   single inference slot is held (``asyncio.wait_for``).
2. The LLM creation path must inject an explicit ``timeout`` into the
   ``DeepThinkingChatOllama`` constructor so the underlying Ollama HTTP client
   does not block indefinitely.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from common.config import OLLAMA_TIMEOUT
from common.exceptions import LLMInferenceError
from core.resource_manager import get_resource_manager


@pytest.fixture(autouse=True)
def restore_semaphore():
    """Restores the inference semaphore after each test."""
    rc = get_resource_manager()
    original = rc.inference_semaphore
    yield
    rc.inference_semaphore = original


@pytest.mark.asyncio
async def test_acquire_inference_lock_times_out_when_slot_held():
    """Second acquisition must raise LLMInferenceError after the short timeout."""
    rc = get_resource_manager()
    rc.inference_semaphore = asyncio.Semaphore(1)

    # 점유: 단일 슬롯을 먼저 획득한다.
    await rc.acquire_inference_lock()
    try:
        with pytest.raises(LLMInferenceError) as excinfo:
            await rc.acquire_inference_lock(timeout=0.05)
        assert excinfo.value.details.get("reason") == "timeout"
    finally:
        rc.release_inference_lock()


@pytest.mark.asyncio
async def test_inference_session_times_out_when_slot_held():
    """inference_session() must propagate the timeout exception on entry."""
    rc = get_resource_manager()
    rc.inference_semaphore = asyncio.Semaphore(1)

    async with rc.inference_session():
        with pytest.raises(LLMInferenceError) as excinfo:
            async with rc.inference_session(timeout=0.05):
                pass  # pragma: no cover - 진입 자체가 실패해야 한다
        assert excinfo.value.details.get("reason") == "timeout"


def test_load_llm_passes_timeout_to_chat_ollama(monkeypatch):
    """LLM 생성 경로가 DeepThinkingChatOllama 생성자에 timeout을 전달해야 한다."""
    monkeypatch.delenv("IS_CI_TEST", raising=False)
    monkeypatch.delenv("IS_UNIT_TEST", raising=False)

    from core import custom_ollama
    from core.model_loader import load_llm

    mock_cls = MagicMock()
    with patch.object(custom_ollama, "DeepThinkingChatOllama", mock_cls):
        load_llm("test-inference-timeout-model")

    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert "timeout" in kwargs, (
        "DeepThinkingChatOllama must receive an explicit timeout"
    )
    assert kwargs["timeout"] == OLLAMA_TIMEOUT
    assert isinstance(kwargs["timeout"], (int, float))
    assert kwargs["timeout"] > 0
