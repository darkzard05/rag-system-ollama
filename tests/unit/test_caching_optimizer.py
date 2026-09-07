import asyncio

import numpy as np
import pytest

from common.utils import fast_hash
from services.optimization.caching_optimizer import DiskCache, SemanticCache


class _FakeEmbedder:
    """Deterministic embedding model for SemanticCache tests.

    Returns the same normalized vector for the same text so that
    exact-key lookups always match above any threshold.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._rng = np.random.default_rng(0)
        self._vectors: dict[str, np.ndarray] = {}

    async def embed_query(self, text: str) -> np.ndarray:
        if text not in self._vectors:
            vec = self._rng.normal(size=self._dim)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._vectors[text] = vec
        return self._vectors[text]


class _BoomEmbedder:
    """Embedder whose embed_query fails loudly if ever invoked.

    Lets SemanticCache tests prove that the provided ``query_embedding``
    is used instead of calling ``_embed`` (B10 passthrough contract).
    """

    def __init__(self) -> None:
        self.called = False

    async def embed_query(self, text: str) -> np.ndarray:
        self.called = True
        raise AssertionError("embed_query must not be called")


def make_query_embedding(dim: int = 64) -> list[float]:
    """Deterministic unit-norm embedding vector for SemanticCache tests."""
    rng = np.random.default_rng(0)
    vec = rng.normal(size=dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _mock_manager(monkeypatch, secret=None, verify_ok=True):
    """Mock security manager to always trust paths and permissions."""
    from unittest.mock import MagicMock

    mock_manager = MagicMock()
    mock_manager.full_verification.return_value = (
        (True, None) if verify_ok else (False, "HMAC mismatch")
    )
    mock_manager.verify_cache_trust.return_value = True
    mock_manager.check_file_permissions.return_value = True
    mock_manager.hmac_secret = secret

    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager
    )
    return mock_manager


@pytest.mark.asyncio
async def test_disk_cache_complex_object(tmp_path, monkeypatch):
    _mock_manager(monkeypatch)
    disk_cache = DiskCache(cache_dir=str(tmp_path))

    key = "test_complex"
    value = {
        "text": "Hello World",
        "numbers": [1, 2, 3],
        "vector": np.array([0.1, 0.2, 0.3]),
        "nested": {"a": 1, "b": {"c": 2}},
    }

    await disk_cache.set(key, value)
    retrieved = await disk_cache.get(key)

    assert retrieved is not None
    assert retrieved["text"] == "Hello World"
    assert retrieved["numbers"] == [1, 2, 3]
    np.testing.assert_array_equal(retrieved["vector"], value["vector"])
    assert retrieved["nested"]["b"]["c"] == 2


@pytest.mark.asyncio
async def test_disk_cache_hmac_verification(tmp_path, monkeypatch):
    secret = "this_is_a_very_secret_key_at_least_32_chars"
    _mock_manager(monkeypatch, secret=secret)

    disk_cache = DiskCache(cache_dir=str(tmp_path))

    key = "hmac_test"
    value = "secure_value"
    await disk_cache.set(key, value)

    assert await disk_cache.get(key) == value

    # Change the secret and try to retrieve
    wrong_secret = "another_very_secret_key_at_least_32_chars"
    _mock_manager(monkeypatch, secret=wrong_secret, verify_ok=False)

    disk_cache_wrong = DiskCache(cache_dir=str(tmp_path))

    assert await disk_cache_wrong.get(key) is None


@pytest.mark.asyncio
async def test_disk_cache_expiration(tmp_path, monkeypatch):
    _mock_manager(monkeypatch)
    disk_cache = DiskCache(cache_dir=str(tmp_path))

    key = "expire_test"
    value = "expire_me"
    await disk_cache.set(key, value, ttl_seconds=0.1)

    assert await disk_cache.get(key) == value

    await asyncio.sleep(0.2)

    assert await disk_cache.get(key) is None


@pytest.mark.asyncio
async def test_semantic_cache_non_expired_entry_returned():
    cache = SemanticCache(embedding_model=_FakeEmbedder(), similarity_threshold=0.95)

    query = "hello world"
    embedding = make_query_embedding()
    await cache.set(query, "stored_value", ttl_seconds=60, query_embedding=embedding)

    cached_key = fast_hash(query)
    assert cached_key in cache.cache
    assert cached_key in cache.embeddings

    result = await cache.get(query, query_embedding=embedding)
    assert result == "stored_value"


@pytest.mark.asyncio
async def test_semantic_cache_expired_entry_not_returned():
    cache = SemanticCache(embedding_model=_FakeEmbedder(), similarity_threshold=0.95)

    query = "hello world"
    embedding = make_query_embedding()
    await cache.set(query, "stored_value", ttl_seconds=1, query_embedding=embedding)

    cached_key = fast_hash(query)
    assert cached_key in cache.cache
    assert cached_key in cache.embeddings

    await asyncio.sleep(1.1)

    result = await cache.get(query, query_embedding=embedding)
    assert result is None
    assert cached_key not in cache.cache
    assert cached_key not in cache.embeddings


@pytest.mark.asyncio
async def test_get_embeds_internally_when_query_embedding_none():
    cache = SemanticCache(embedding_model=_FakeEmbedder(), similarity_threshold=0.95)

    await cache.set("q", "stored_value")

    result = await cache.get("q")

    assert result == "stored_value"


@pytest.mark.asyncio
async def test_get_uses_provided_query_embedding_without_embedding_call():
    embedder = _BoomEmbedder()
    cache = SemanticCache(embedding_model=embedder, similarity_threshold=0.95)

    v1 = make_query_embedding()
    await cache.set("q", "stored_value", query_embedding=v1)

    result = await cache.get("q", query_embedding=v1)

    assert result == "stored_value"
    assert not embedder.called


@pytest.mark.asyncio
async def test_set_embeds_internally_when_query_embedding_none():
    cache = SemanticCache(embedding_model=_FakeEmbedder(), similarity_threshold=0.95)

    await cache.set("q", "value")

    assert cache.cache_size == 1
    assert await cache.get("q") == "value"


@pytest.mark.asyncio
async def test_set_uses_provided_query_embedding():
    embedder = _BoomEmbedder()
    cache = SemanticCache(embedding_model=embedder, similarity_threshold=0.95)

    v = make_query_embedding()
    await cache.set("q", "value", query_embedding=v)

    assert cache.cache_size == 1
    assert await cache.get("q", query_embedding=v) == "value"
    assert not embedder.called
