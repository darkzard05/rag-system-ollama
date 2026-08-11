"""
P2 검증: _preload_model의 LLM 실질 워밍 추론.

프리로드 step3(get_llm)가 단순 클라이언트 생성에 그치면 첫 쿼리가 Ollama 콜드
모델 로드(~20s)를 지불하므로, 실질 워밍 추론(invoke("warmup"))이 정확히 1회
발생하는지 검증합니다.

헤르메틱: get_embedder/get_flashranker/get_llm 3개를 모두 스텁 — 실 Ollama/
모델/네트워크 호출 없이 CI(Ollama 부재)에서도 green이어야 합니다.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import core.pipeline_builder as pb
from core.model_loader import ModelManager


class _RecordingLLM:
    """invoke 호출을 기록하는 동기 fake (프리로드는 to_thread로 실행)."""

    def __init__(self) -> None:
        self.invoke_calls: list[tuple[str, dict]] = []

    def invoke(self, prompt: str, **kwargs) -> str:
        self.invoke_calls.append((prompt, kwargs))
        return "warmup ok"


class _FakeEmbedder:
    """embed_query가 즉시 리스트를 반환하는 동기 fake (to_thread로 실행)."""

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


def _patch_model_manager(monkeypatch, llm_fake) -> None:
    """step 1(임베딩)/step 2(FlashRank)/step 3(LLM)을 모두 스텁합니다."""
    monkeypatch.setattr(
        ModelManager,
        "get_embedder",
        AsyncMock(return_value=_FakeEmbedder()),
    )
    monkeypatch.setattr(
        ModelManager,
        "get_flashranker",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(ModelManager, "get_llm", AsyncMock(return_value=llm_fake))


@pytest.mark.asyncio
async def test_preload_warmup_invokes_llm_exactly_once_with_warmup(monkeypatch):
    """프리로드 step3에서 LLM.invoke가 정확히 1회, "warmup" 인자로 호출되는지 검증합니다."""
    recording_llm = _RecordingLLM()
    _patch_model_manager(monkeypatch, recording_llm)

    await pb._preload_model()

    assert len(recording_llm.invoke_calls) == 1
    prompt, _kwargs = recording_llm.invoke_calls[0]
    assert prompt == "warmup"


@pytest.mark.asyncio
async def test_preload_warmup_swallows_get_llm_failure(monkeypatch):
    """get_llm이 예외를 던져도 _preload_model이 재발생 없이 조용히 종료되는지 검증합니다."""
    _patch_model_manager(monkeypatch, llm_fake=MagicMock())
    monkeypatch.setattr(
        ModelManager,
        "get_llm",
        AsyncMock(side_effect=RuntimeError("ollama down")),
    )

    # 재발생하면 테스트가 실패합니다.
    await pb._preload_model()
