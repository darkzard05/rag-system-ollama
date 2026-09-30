# CoordCache Thread Flatness Measurement

**Date:** 2026-09-26 14:27:45
**Task:** Wave-0 Task 0 — Verify CoordCache background entities stay flat across 5 sequential queries
**Ollama backend:** Available (`qwen3:4b-instruct-2507-q4_K_M`, `nomic-embed-text-v2-moe:latest`)

---

## Source Code Verification

### Start Guards (`src/cache/coord_cache.py` L114-152)

`_ensure_owner_loop()` implements a **double-check pattern** with `_owner_start_lock`:

1. **First check (L116-122):** If `_owner_thread is not None and is_alive() and _owner_loop is not None and is_running()` → return immediately (no lock contention).
2. **Second check (L124-131):** Same condition inside `_owner_start_lock` → return (race-safe).
3. **Only if both fail** does it create a new `CoordCacheLoop` thread (L142-145).

This confirms: **per-query spawn is forbidden by design.**

### `_stop_owner_loop` Log Emission (`src/cache/coord_cache.py` L170-186)

`_stop_owner_loop()` contains **zero logger calls**. Grep across `src/` found only 2 matches for `_stop_owner_loop`: the definition at L170 and the call at L507 (inside `close()`). Neither contains a `logger.info/debug/error` line. This confirms: **`_stop_owner_loop` emits no log by design.**

### `close()` Callers (`src/cache/coord_cache.py` L497-514)

Grep for `coord_cache.close`, `CoordCacheManager.*\.close`, and `cache.*\.close\(\)` across all of `src/` returned **zero matches**. The only `.close()` calls in `src/` are on `_conn`, `doc`, and `loop` objects — none on `CoordCacheManager`. This confirms: **`close()` has zero callers in src/.**

### Flag Definitions (`src/cache/coord_cache.py` L55-89)

- `_worker_task: asyncio.Task | None` — initialized to `None` (L83), set once in `_ensure_worker_started` (L203-204), reset to `None` in `close()` (L529).
- `_eviction_started: bool` — initialized to `False` (L86), set to `True` once in `_ensure_schema` (L196-197), reset to `False` in `close()` (L514).
- `_eviction_task` — created once in `_ensure_schema` (L198), cancelled in `close()` (L524-526).

---

## Measurement Results

### 5-Row Table

| Query # | CoordCacheLoop thread count (`threading.enumerate` filter) | `_owner_thread.is_alive()` | `_worker_task is None` | `_eviction_started` | Start-log lines seen |
|---------|-----------------------------------------------------------|---------------------------|----------------------|--------------------|--------------------|
| 1       | 1                                                         | True                      | False                | False              | 2                  |
| 2       | 1                                                         | True                      | False                | False              | 0                  |
| 3       | 1                                                         | True                      | False                | False              | 0                  |
| 4       | 1                                                         | True                      | False                | False              | 0                  |
| 5       | 1                                                         | True                      | False                | False              | 0                  |

### Exact Flag Values Per Round

- **Round 1:** `coord_loop_count=1`, `owner_thread_alive=True`, `owner_thread_is_none=False`, `owner_loop_is_running=True`, `worker_task_is_none=False`, `worker_task_done=False`, `eviction_started=False`, `eviction_task_is_none=True`, `eviction_task_done=None`
- **Round 2:** Identical to Round 1 (all flags unchanged)
- **Round 3:** Identical to Round 1 (all flags unchanged)
- **Round 4:** Identical to Round 1 (all flags unchanged)
- **Round 5:** Identical to Round 1 (all flags unchanged)

### Log Lines Per Round

- **Round 1:** 2 lines — `INFO:cache.coord_cache:좌표 캐시 owner 이벤트 루프 시작됨`, `INFO:cache.coord_cache:좌표 캐시 백그라운드 워커 시작됨`
- **Rounds 2-5:** 0 lines each

---

## Verdict

**VERDICT: flat**

The CoordCache background entities remain flat across all 5 sequential queries. The `CoordCacheLoop` thread count is exactly 1 in every round. No new threads are spawned on subsequent queries. The `_owner_thread.is_alive()` is `True` throughout. The `_worker_task` and `_eviction_started` flags do not change. Zero log lines are emitted after Round 1, confirming that `_ensure_owner_loop`'s double-check guards prevent per-query spawn and that `_stop_owner_loop` emits no log by design.

---

## Log Excerpt Proving Start-Without-Stop by Design

The only log lines observed across all 5 rounds were emitted in Round 1:

```
INFO:cache.coord_cache:좌표 캐시 owner 이벤트 루프 시작됨
INFO:cache.coord_cache:좌표 캐시 백그라운드 워커 시작됨
```

Rounds 2-5 produced **zero** log lines. This is consistent with the source code:

1. `_ensure_owner_loop()` returns early on the first check (L116-122) because `_owner_thread is not None and is_alive()` — no `logger.info` call on the early return path.
2. `_ensure_worker_started()` returns early because `_worker_task is not None` (L203) — no `logger.info` call on the early return path.
3. `_stop_owner_loop()` (L170-186) has **no logger calls** — confirmed by grep.
4. `close()` has **zero callers** in `src/` — confirmed by grep.

Therefore, the start-without-stop pattern is by design: the owner loop starts once on the first query and never stops, because `close()` is never called in the normal code path.

---

## Measurement Script

The throwaway measurement script is at `reports/measure_coord_threads_20260926_142745.py`. It imports `CoordCacheManager` via `from cache.coord_cache import coord_cache`, issues 5 sequential `save_coords` calls via `asyncio.run`, and dumps `threading.enumerate()` filtered for `CoordCacheLoop`, all flag values, and captured log lines per round.
