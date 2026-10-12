"""Repro: SemanticCache file_hash namespace bypass (Defect 1).

``f"{file_hash}:{query}"`` keys from different documents must never
collide. The real SemanticCache only checks cosine similarity, so a
prefix-insensitive embedder (like real Ollama/nomic embeddings where a
64-hex prefix is noise) returns the other document's answer.
"""

import numpy as np
import pytest

from services.optimization.backends import SemanticCache


class PrefixInsensitiveEmbedder:
    """Emulates a real semantic embedder: hex prefix ignored, query body rules."""

    async def embed_query(self, text: str) -> list[float]:
        query = text.split(":", 1)[1] if ":" in text else text
        vec = np.zeros(16)
        vec[abs(hash(query)) % 16] = 1.0
        return vec.tolist()


@pytest.mark.asyncio
async def test_cross_document_same_query_must_miss() -> None:
    cache = SemanticCache(
        embedding_model=PrefixInsensitiveEmbedder(),
        similarity_threshold=0.95,
    )
    hash_a = "a" * 64
    hash_b = "b" * 64
    query = "이 문서의 수도는?"
    await cache.set(f"{hash_a}:{query}", {"response": "문서A 답", "confidence": 0.99})
    hit = await cache.get(f"{hash_b}:{query}")
    assert hit is None, f"교차 문서 히트 발생 (격리 위반): {hit!r}"
