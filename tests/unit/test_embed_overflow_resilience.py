"""Ollama 임베딩 컨텍스트 초과(400 에러) 자동 폴백 복원력 단위 테스트.

계획 §2 (TDD Red-to-Green Strategy):
  - _NoTruncateOllamaEmbeddings 에 명백히 긴 텍스트를 주입하고,
  - 1차 호출(truncate=False)에서 ollama.ResponseError(400, "exceeds the context length") 발생 시
  - 자동으로 truncate=True로 2차 시도하여 정상 벡터를 반환하는지 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import ollama
import pytest

from core.model_loader import _NoTruncateOllamaEmbeddings


def test_embed_overflow_fallback_on_400() -> None:
    embedder = _NoTruncateOllamaEmbeddings(model="test-model", base_url="http://fake")

    call_count = 0

    def mock_embed(model, text_or_texts, **kwargs):
        nonlocal call_count
        call_count += 1
        if not kwargs.get("truncate", False):
            resp = MagicMock()
            resp.status_code = 400
            raise ollama.ResponseError(
                "the input length exceeds the context length", resp.status_code
            )
        return {"embeddings": [[0.1] * 768]}

    mock_client = MagicMock()
    mock_client.embed = mock_embed
    embedder._client = mock_client

    res = embedder.embed_documents(["아주 긴 텍스트..." * 500])
    assert len(res) == 1
    assert len(res[0]) == 768
    assert call_count == 2  # 1차 실패(truncate=False) -> 2차 성공(truncate=True)


def test_non_overflow_400_raises_immediately() -> None:
    embedder = _NoTruncateOllamaEmbeddings(model="test-model", base_url="http://fake")

    def mock_embed(model, text_or_texts, **kwargs):
        resp = MagicMock()
        resp.status_code = 400
        raise ollama.ResponseError("bad request other reason", resp.status_code)

    mock_client = MagicMock()
    mock_client.embed = mock_embed
    embedder._client = mock_client

    with pytest.raises(ollama.ResponseError, match="bad request"):
        embedder.embed_documents(["normal text"])


def test_is_ollama_embedder_unwraps_wrapper() -> None:
    """MemoizingEmbedding 등 래퍼 내부의 OllamaEmbeddings를 정확히 판별한다."""
    from langchain_ollama import OllamaEmbeddings
    from core.semantic_chunker import _is_ollama_embedder

    ollama_inst = OllamaEmbeddings(model="test")

    # 래퍼 목 객체 생성
    class MockWrapper:
        def __init__(self, inner):
            self._inner = inner

    wrapped = MockWrapper(MockWrapper(ollama_inst))
    assert _is_ollama_embedder(wrapped) is True
    assert _is_ollama_embedder(MagicMock()) is False
