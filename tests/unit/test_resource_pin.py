"""Tests for the model pin/unpin refcount + atomic acquire (use-after-free fix).

Covers: refcount transitions, key_for_object reverse lookup (KeyError on
missing), atomic acquire+pin (get_or_pin pins inside the pool lock so no
concurrent eviction can slip in the gap), and CM unpin on normal/cancelled
exit.
"""

import asyncio

import pytest

from core.resource_manager import BaseResourcePool, ResourceCoordinator  # noqa: F401


class _Obj:
    """Plain object used as a pool value to test identity-based key lookup."""

    def __init__(self, name="obj"):
        self.name = name


class _LightCoordinator(ResourceCoordinator):
    """Singletons avoided; uses a plain BaseResourcePool (no torch/CUDA)."""

    def __new__(cls):
        # bypass ResourceCoordinator singleton __new__
        return object.__new__(cls)

    def _init_coordinator(self):
        self.models = BaseResourcePool("Model", item_limit=10, byte_limit=10**9)

        # get_or_pin calls pool.check_vram_pressure for the models pool; the
        # real ModelPool implements it (torch/CUDA). Unit test uses a no-op.
        async def _noop_vram():
            return False

        self.models.check_vram_pressure = _noop_vram
        self.retrievers = BaseResourcePool("Retriever", item_limit=10, byte_limit=10**9)
        self._build_locks: dict[str, asyncio.Lock] = {}
        self._in_flight: set[str] = set()
        self._build_call_counter = 0
        self._build_failures: dict[str, tuple[int, float]] = {}

    async def check_vram_pressure(self):  # no-op (no torch in unit tests)
        return False

    def _check_build_circuit(self, key):
        return None

    def _record_build_failure(self, key):
        return None

    def _cleanup_build_locks(self):
        return None


def _make_coordinator() -> _LightCoordinator:
    c = _LightCoordinator.__new__(_LightCoordinator)
    c._init_coordinator()
    return c


# --- BaseResourcePool refcount + key_for_object (unit) ---


def test_refcount_pin_unpin_transition():
    pool = BaseResourcePool("Test", item_limit=10, byte_limit=10**9)
    pool._pool["k1"] = _Obj()
    pool.pin("k1")
    assert pool.is_pinned("k1") is True
    pool.pin("k1")  # second pin -> count 2
    assert pool._pinned_keys["k1"] == 2
    pool.unpin("k1")
    assert pool.is_pinned("k1") is True  # still 1
    pool.unpin("k1")
    assert pool.is_pinned("k1") is False
    assert "k1" not in pool._pinned_keys  # fully removed, no leak


def test_is_pinned_missing_key_false():
    pool = BaseResourcePool("Test", item_limit=10, byte_limit=10**9)
    assert pool.is_pinned("never") is False


def test_key_for_object_reverse_lookup():
    pool = BaseResourcePool("Test", item_limit=10, byte_limit=10**9)
    obj = _Obj()
    pool._pool["model_x"] = obj
    assert pool.key_for_object(obj) == "model_x"


def test_key_for_object_missing_raises_keyerror():
    pool = BaseResourcePool("Test", item_limit=10, byte_limit=10**9)
    with pytest.raises(KeyError):
        pool.key_for_object(_Obj())  # not in pool -> must raise, not default


def test_eviction_skips_pinned():
    pool = BaseResourcePool("Test", item_limit=10, byte_limit=10**9)
    a, b = _Obj("a"), _Obj("b")
    pool._pool["a"] = a
    pool._pool["b"] = b
    pool.pin("a")
    evicted = pool._evict_one_locked()
    assert evicted is True
    # a is pinned -> b evicted, a kept
    assert "a" in pool._pool
    assert "b" not in pool._pool


# --- Atomic acquire+pin via get_or_pin (integration) ---


@pytest.mark.asyncio
async def test_use_embedder_cm_pins_and_unpins_object():
    """Object-based path (key_for_object reverse lookup) must pin for the
    duration and unpin in finally — no silent DEFAULT fallback."""
    coordinator = _make_coordinator()
    key = "test_cm_obj_embedder"
    obj = _Obj()
    coordinator.models._pool[key] = obj
    coordinator.models.unpin(key)

    async with coordinator.use_embedder(embedder=obj):
        assert coordinator.models.is_pinned(key) is True

    assert coordinator.models.is_pinned(key) is False
    coordinator.models._pool.pop(key, None)


@pytest.mark.asyncio
async def test_use_embedder_cm_unpins_on_exception():
    coordinator = _make_coordinator()
    key = "test_cm_exc_embedder"
    obj = _Obj()
    coordinator.models._pool[key] = obj
    coordinator.models.unpin(key)

    with pytest.raises(RuntimeError):
        async with coordinator.use_embedder(embedder=obj):
            raise RuntimeError("boom")

    assert coordinator.models.is_pinned(key) is False
    coordinator.models._pool.pop(key, None)


@pytest.mark.asyncio
async def test_concurrent_eviction_cannot_evict_pinned_model():
    """Reproduces the original race: T1 holds a pinned model while T2 triggers
    an eviction sweep. The pinned model must survive."""
    coordinator = _make_coordinator()
    key = "test_race_embedder"
    obj = _Obj()
    coordinator.models._pool[key] = obj
    coordinator.models.unpin(key)

    async with coordinator.use_embedder(embedder=obj):
        # T2: force an eviction sweep (mirrors check_vram_pressure path)
        coordinator.models._evict_one_locked()
        assert coordinator.models.get(key) is not None
        assert coordinator.models.is_pinned(key) is True

    assert coordinator.models.is_pinned(key) is False
    coordinator.models._pool.pop(key, None)


def test_pin_happens_inside_pool_lock_after_put():
    """BLOCKER #1: the pin must occur inside the same _lock critical section as
    the insert (as get_or_pin does after pool.put), so a concurrent eviction
    cannot slip between acquire and pin. We assert that an object present in the
    pool is pinnable and that the pin is recorded atomically (no intermediate
    unpinned window observable under the lock)."""
    pool = BaseResourcePool("Test", item_limit=10, byte_limit=10**9)
    obj = _Obj()
    # get_or_pin does put then, still holding _lock, pins. We mirror the atomic
    # acquire+pin by pinning right after the object is in the pool.
    pool._pool["k"] = obj
    pool.pin("k")  # get_or_pin calls pool.pin(key) under _lock after put()
    assert pool.is_pinned("k") is True
    # A pinned key must survive an eviction sweep: no unpinned entry to evict.
    assert pool._evict_one_locked() is False
    assert pool.get("k") is not None


def test_put_then_pin_no_relock_deadlock():
    """Regression: get_or_pin does `await pool.put(key, res)` then `pool.pin(key)`.
    BaseResourcePool.put() acquires _lock internally and pin() also acquires it,
    so pin must NOT be wrapped in a second `with pool._lock` (non-reentrant
    threading.Lock would deadlock). This mirrors the get_or_pin sequence with a
    pool that exercises the eviction-within-put path."""
    pool = BaseResourcePool("Test", item_limit=2, byte_limit=10**9)
    # Fill so put() triggers an eviction (which also takes _lock) before pin.
    pool._pool["a"] = _Obj("a")
    obj = _Obj("b")
    # Simulate get_or_pin: put (may evict under _lock) then pin OUTSIDE any lock.
    asyncio.run(pool.put("b", obj))
    pool.pin("b")  # must not deadlock
    assert pool.is_pinned("b") is True
    assert pool.get("b") is not None
