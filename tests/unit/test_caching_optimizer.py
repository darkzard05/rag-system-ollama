import asyncio

import numpy as np
import pytest

from services.optimization.caching_optimizer import DiskCache


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
