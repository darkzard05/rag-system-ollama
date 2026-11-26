"""Regression: a paraphrased query must not reuse another model's answer.

The query-cache key carries both the document hash and the LLM model. The
semantic cache must isolate on both, not just on the document hash. Exact-key
lookups already include the model; this test pins the similarity path, where a
model switch previously returned the old model's answer with no sources.
"""

import numpy as np
import pytest

from core.graph._graph_core import _query_cache_key
from services.optimization.backends import SemanticCache


class _KeywordEmbedder:
    """Namespace prefixes are noise; only the query body drives similarity."""

    async def embed_query(self, text: str) -> list[float]:
        query = text.split(":", 2)[-1]
        vec = np.zeros(4)
        vec[0 if "CM3" in query else 1] = 1.0
        return vec.tolist()


@pytest.mark.asyncio
async def test_model_switch_does_not_reuse_semantic_answer() -> None:
    cache = SemanticCache(embedding_model=_KeywordEmbedder(), similarity_threshold=0.9)
    file_hash = "a" * 64
    original = {"response": "ANSWER FROM MODEL A"}
    await cache.set(_query_cache_key(file_hash, "what is CM3", "model-a"), original)

    other_model = await cache.get(
        _query_cache_key(file_hash, "what is CM3 (paraphrase)", "model-b")
    )
    assert other_model is None

    same_model = await cache.get(
        _query_cache_key(file_hash, "what is CM3 (paraphrase)", "model-a")
    )
    assert same_model == original
