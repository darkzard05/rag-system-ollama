"""Colons in Ollama model names must not collapse the query-cache namespace."""

from __future__ import annotations

import pytest

from core.graph._graph_core import _query_cache_key
from services.optimization.backends import SemanticCache

FILE_HASH = "a" * 64
MODEL_A = "qwen3:4b-instruct-2507-q4_K_M"
MODEL_B = "qwen3:8b-instruct-2507-q4_K_M"
QUERY = "이 문서의 핵심 내용은 무엇인가요"
VECTOR = [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_colon_model_names_do_not_share_cached_answer() -> None:
    cache = SemanticCache(similarity_threshold=0.92)
    answer = {"response": "ANSWER FROM MODEL A"}
    await cache.set(
        _query_cache_key(FILE_HASH, QUERY, MODEL_A),
        answer,
        query_embedding=VECTOR,
    )

    other_model = await cache.get(
        _query_cache_key(FILE_HASH, QUERY, MODEL_B),
        query_embedding=VECTOR,
    )

    assert other_model is None
    same_model = await cache.get(
        _query_cache_key(FILE_HASH, QUERY, MODEL_A),
        query_embedding=VECTOR,
    )
    assert same_model == answer
