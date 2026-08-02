import os
import shutil

import numpy as np
import pytest

from services.optimization.caching_optimizer import DiskCache


@pytest.fixture
def disk_cache():
    cache_dir = "./.test_cache"
    cache = DiskCache(cache_dir=cache_dir)
    yield cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


@pytest.mark.asyncio
async def test_disk_cache_complex_object(disk_cache, monkeypatch):
    # Mock security manager to always trust paths and permissions
    from unittest.mock import MagicMock

    mock_manager = MagicMock()
    mock_manager.full_verification.return_value = (True, None)
    mock_manager.verify_cache_trust.return_value = True
    mock_manager.check_file_permissions.return_value = True
    mock_manager.hmac_secret = None

    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager
    )

    # Re-initialize DiskCache to ensure it uses the mocked manager
    disk_cache = DiskCache(cache_dir="./.test_cache")

    key = "test_complex"
    # Complex object: dict with list, numpy array, and nested dict
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
async def test_disk_cache_hmac_verification(disk_cache, monkeypatch):
    # Mock security manager to trust all paths and permissions
    from unittest.mock import MagicMock

    secret = "this_is_a_very_secret_key_at_least_32_chars"
    mock_manager = MagicMock()
    mock_manager.full_verification.return_value = (True, None)
    mock_manager.verify_cache_trust.return_value = True
    mock_manager.check_file_permissions.return_value = True
    mock_manager.hmac_secret = secret

    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager
    )

    # Re-initialize DiskCache to use the mocked manager
    disk_cache = DiskCache(cache_dir="./.test_cache_hmac")

    key = "hmac_test"
    value = "secure_value"
    await disk_cache.set(key, value)

    # Should retrieve correctly with correct key
    assert await disk_cache.get(key) == value

    # Now change the secret and try to retrieve
    wrong_secret = "another_very_secret_key_at_least_32_chars"
    mock_manager_wrong = MagicMock()
    mock_manager_wrong.verify_cache_trust.return_value = True
    mock_manager_wrong.check_file_permissions.return_value = True
    mock_manager_wrong.hmac_secret = wrong_secret

    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager_wrong
    )

    # Re-initialize DiskCache to use the wrong manager
    disk_cache_wrong = DiskCache(cache_dir="./.test_cache_hmac")

    # Should fail to retrieve due to HMAC mismatch
    assert await disk_cache_wrong.get(key) is None

    if os.path.exists("./.test_cache_hmac"):
        shutil.rmtree("./.test_cache_hmac")


@pytest.mark.asyncio
async def test_disk_cache_expiration(disk_cache, monkeypatch):
    # Mock security manager to trust all paths
    from unittest.mock import MagicMock

    mock_manager = MagicMock()
    mock_manager.full_verification.return_value = (True, None)
    mock_manager.verify_cache_trust.return_value = True
    mock_manager.check_file_permissions.return_value = True
    mock_manager.hmac_secret = None

    monkeypatch.setattr(
        "security.cache_security.get_security_manager", lambda: mock_manager
    )

    # Re-initialize DiskCache to use the mocked manager
    disk_cache = DiskCache(cache_dir="./.test_cache_exp")

    key = "expire_test"
    value = "expire_me"
    # Set TTL to 0.1 seconds
    await disk_cache.set(key, value, ttl_seconds=0.1)

    # Immediate retrieval should work
    assert await disk_cache.get(key) == value

    # Wait for expiration
    import asyncio

    await asyncio.sleep(0.2)

    # Should be expired
    assert await disk_cache.get(key) is None

    if os.path.exists("./.test_cache_exp"):
        shutil.rmtree("./.test_cache_exp")
