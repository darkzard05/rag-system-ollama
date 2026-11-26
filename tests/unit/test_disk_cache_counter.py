"""DiskCache must not count an entry whose file was never written."""

from unittest.mock import patch

import pytest

from services.optimization.backends import DiskCache


@pytest.mark.asyncio
async def test_failed_write_does_not_increase_cache_size(tmp_path) -> None:
    cache = DiskCache(cache_dir=str(tmp_path))
    real_open = open

    def _failing_open(file, *args, **kwargs):
        if str(file).endswith(".cache"):
            raise OSError("disk full")
        return real_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=_failing_open):
        await cache.set("key", {"value": 1})

    assert cache.get_stats().cache_size == 0
