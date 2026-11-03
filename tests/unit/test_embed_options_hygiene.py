"""Ollama 임베딩 요청 options 정리 단위 테스트.

langchain-ollama 1.x는 임베딩 요청에 샘플러 옵션(mirostat 등)을 None 값으로도
포함시켜 Ollama 0.40+에서 ``invalid option provided`` 경고를 유발한다.
``_NoTruncateOllamaEmbeddings``가 이를 제거하는지 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from core.model_loader import _NoTruncateOllamaEmbeddings

_UNSUPPORTED = frozenset({"mirostat", "mirostat_eta", "mirostat_tau", "tfs_z"})


def test_default_params_drop_unsupported_sampler_options() -> None:
    embedder = _NoTruncateOllamaEmbeddings(
        model="test-model", base_url="http://fake", temperature=0.3
    )

    params = embedder._default_params

    assert _UNSUPPORTED.isdisjoint(params)
    assert all(value is not None for value in params.values())
    assert params["temperature"] == 0.3


def test_embed_request_passes_clean_options() -> None:
    embedder = _NoTruncateOllamaEmbeddings(model="test-model", base_url="http://fake")
    captured: dict[str, object] = {}

    def mock_embed(model, texts, **kwargs):
        captured.update(kwargs)
        return {"embeddings": [[0.1, 0.2]]}

    mock_client = MagicMock()
    mock_client.embed = mock_embed
    embedder._client = mock_client

    embedder.embed_documents(["hello"])

    options = captured["options"]
    assert isinstance(options, dict)
    assert _UNSUPPORTED.isdisjoint(options)
    assert all(value is not None for value in options.values())
