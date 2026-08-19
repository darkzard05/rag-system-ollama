"""
F1 regression + characterization tests: decouple Streamlit from core.session.manager.

These tests prove two things across the refactor:

1. BASELINE (pre-fix behavior, captured on the UNMODIFIED module):
   With ``streamlit`` unimportable, ``import core.session.manager`` raised
   ImportError because the module imported ``streamlit as st`` at top level.
   This is pinned by ``test_baseline_top_level_streamlit_import_absent``: on the
   original code the top-level ``import streamlit as st`` was present (so that
   assertion was RED), and after the fix it is gone (so the assertion is GREEN).

2. POST-FIX behavior (the regression guard):
   - ``import core.session.manager`` must succeed even when ``streamlit`` is
     unimportable (UI logic moved out of core).
   - ``SessionManager.get`` must fall back to the global-store value when no
     UI-sync adapter is installed (``set_ui_sync(None)``), without raising.
   - UI writes are routed through the pluggable adapter (mirroring the exact
     list/dict/copy semantics that previously lived in ``sync_to_streamlit``).
"""

import os
import sys
from typing import Any

import pytest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class _BlockStreamlit:
    """meta_path finder that makes ``streamlit`` (and submodules) unimportable."""

    def find_spec(self, name, path=None, target=None):  # noqa: ANN001, D401
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError(f"streamlit is blocked for test: {name}")
        return None


class _no_streamlit:
    def __enter__(self) -> None:
        sys.modules.pop("streamlit", None)
        for mod in list(sys.modules):
            if mod == "streamlit" or mod.startswith("streamlit."):
                sys.modules.pop(mod, None)
        sys.meta_path.insert(0, _BlockStreamlit())

    def __exit__(self, *exc: object) -> None:
        sys.meta_path[:] = [
            f for f in sys.meta_path if not isinstance(f, _BlockStreamlit)
        ]
        for mod in list(sys.modules):
            if mod == "streamlit" or mod.startswith("streamlit."):
                sys.modules.pop(mod, None)


def _block_streamlit() -> _no_streamlit:
    return _no_streamlit()


def _snapshot_sys_modules() -> dict[str, object]:
    """Capture every ``core.session*"`` and ``streamlit*"`` module object before we
    mutate ``sys.modules``.

    ``tests/conftest.py`` merges the ``core.*`` / ``src.core.*`` aliases into a
    single authoritative module object at process start. Some tests below pop
    ``core.session`` from ``sys.modules`` to force a fresh re-import; that creates
    a *new* module object and splits the previously-merged ``SessionManager`` class,
    which leaks across tests and breaks downstream consumers. Snapshotting and
    restoring those modules around the mutation keeps the conftest merge intact.
    """
    return {
        name: mod
        for name, mod in sys.modules.items()
        if name.startswith("core.session")
        or name == "streamlit"
        or name.startswith("streamlit.")
    }


def _restore_sys_modules(snapshot: dict[str, object]) -> None:
    """Restore the ``core.session*"`` / ``streamlit*"`` modules to the snapshot.

    Re-applies every captured module object, then drops any freshly-imported
    session/streamlit submodule that was not present in the snapshot.
    """
    for name, mod in snapshot.items():
        sys.modules[name] = mod
    for name in [
        n
        for n in sys.modules
        if n.startswith("core.session") or n.startswith("streamlit.")
    ]:
        if name not in snapshot:
            sys.modules.pop(name, None)


def _module_source_has_top_level_streamlit_import() -> bool:
    # type: () -> bool
    """True if ``src/core/session/manager.py`` still imports streamlit at top level."""
    path = os.path.join(SRC_ROOT, "core", "session", "manager.py")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped == "import streamlit as st":
                return True
    return False


def test_baseline_top_level_streamlit_import_absent() -> None:
    """F1 invariant: core no longer imports streamlit at module top level.

    On the unmodified module this assertion was RED (``import streamlit as st``
    existed at line 21). After the decoupling it is GREEN.
    """
    assert not _module_source_has_top_level_streamlit_import()


def test_core_import_succeeds_with_streamlit_blocked() -> None:
    """F1 regression: importing core.session.manager must not require streamlit."""
    # Snapshot before mutating sys.modules so the conftest module-merge stays
    # intact after the test (prevents cross-test leakage of SessionManager).
    _snap = _snapshot_sys_modules()
    try:
        with _block_streamlit():
            for mod in list(sys.modules):
                if mod.startswith("core.session"):
                    sys.modules.pop(mod, None)
            import core.session.manager as m  # noqa: F401

            assert hasattr(m.SessionManager, "set_ui_sync")
    finally:
        _restore_sys_modules(_snap)


def test_get_falls_back_to_global_store_when_no_adapter() -> None:
    """With no UI adapter installed, get() returns the global-store value."""
    # Snapshot before mutating sys.modules so the conftest module-merge survives
    # the test (prevents cross-test leakage of SessionManager).
    _snap = _snapshot_sys_modules()
    try:
        with _block_streamlit():
            for mod in list(sys.modules):
                if mod.startswith("core.session"):
                    sys.modules.pop(mod, None)
            from core.session import SessionManager

            try:
                SessionManager.reset()
                SessionManager.set_ui_sync(None)
                sid = "f1_fallback_sid"
                SessionManager.set_session_id(sid)
                SessionManager.init_session()
                SessionManager.set("f1_marker", "from_global_store", session_id=sid)

                # No adapter -> must read the global store, never touch st.session_state.
                assert (
                    SessionManager.get("f1_marker", session_id=sid)
                    == "from_global_store"
                )
                # Missing key -> default, no raise.
                assert (
                    SessionManager.get("f1_missing", default="dv", session_id=sid)
                    == "dv"
                )
            finally:
                SessionManager.set_ui_sync(None)
    finally:
        _restore_sys_modules(_snap)


class _FakeSync:
    """Minimal adapter recording write() calls with the exact value passed."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, object]] = []

    def write(self, key: str, val: object) -> None:
        self.writes.append((key, val))

    def read(self, key: str, default: object = None) -> object:
        return default


def test_set_ui_sync_adapter_routes_writes() -> None:
    """sync_to_streamlit must route writes through the installed adapter.

    Runs with streamlit importable (it is, in this env) and stubs the Streamlit
    runtime guard so the sync path executes; we only assert the routing/semantics
    of the writes, not any real UI.
    """
    import streamlit.runtime.scriptrunner as scriptrunner  # type: ignore

    # Snapshot before popping so the conftest module-merge survives the test.
    _snap = _snapshot_sys_modules()
    try:
        for mod in list(sys.modules):
            if mod.startswith("core.session"):
                sys.modules.pop(mod, None)
        from core.session import SessionManager

        fake = _FakeSync()
        real_is_running = SessionManager._is_streamlit_running
        real_ctx = scriptrunner.get_script_run_ctx
        try:
            SessionManager._is_streamlit_running = classmethod(  # type: ignore[assignment]
                lambda cls: True
            )
            scriptrunner.get_script_run_ctx = lambda: object()  # truthy ctx

            SessionManager.reset()
            SessionManager.set_ui_sync(fake)
            sid = "f1_adapter_sid"
            SessionManager.set_session_id(sid)
            SessionManager.init_session()

            # list value must be copied (slice), not aliased.
            SessionManager.set("f1_list", [1, 2, 3], session_id=sid)
            # dict value must be copied.
            SessionManager.set("f1_dict", {"a": 1}, session_id=sid)
            SessionManager.set("f1_scalar", "x", session_id=sid)

            SessionManager.sync_to_streamlit(sid)

            written = dict(fake.writes)
            assert written["f1_list"] == [1, 2, 3]
            assert written["f1_dict"] == {"a": 1}
            assert written["f1_scalar"] == "x"
        finally:
            scriptrunner.get_script_run_ctx = real_ctx
            SessionManager._is_streamlit_running = real_is_running  # type: ignore[assignment]
            SessionManager.set_ui_sync(None)
    finally:
        _restore_sys_modules(_snap)


def test_streamlitsessionsync_mirrors_copy_semantics() -> None:
    """StreamlitSessionSync.write must copy lists/dicts (no UI aliasing)."""
    from unittest.mock import patch

    from ui import session_sync
    from ui.session_sync import StreamlitSessionSync

    # `SessionManager` routes UI writes through the installed adapter, which
    # mirrors state into ``session_sync.st.session_state``. We substitute a plain
    # dict-like store on the adapter's own ``st`` binding (not the global
    # ``streamlit.session_state`` module attribute) so the assertion is robust
    # even when another test has already booted the Streamlit runtime — Streamlit
    # caches an internal ``session_state`` reference that the module attribute
    # patch does not reach.
    store: dict[str, Any] = {}

    class _SessionStateProxy:
        def __getitem__(self, k):
            return store[k]

        def __setitem__(self, k, v):
            store[k] = v

        def get(self, k, default=None):
            return store.get(k, default)

    with patch.object(session_sync.st, "session_state", _SessionStateProxy()):
        original_list = [1, 2, 3]
        original_dict = {"a": 1}

        StreamlitSessionSync.write("k_list", original_list)
        StreamlitSessionSync.write("k_dict", original_dict)
        StreamlitSessionSync.write("k_scalar", "x")

        assert store["k_list"] == original_list
        assert store["k_list"] is not original_list  # sliced copy
        assert store["k_dict"] == original_dict
        assert store["k_dict"] is not original_dict  # shallow copy
        assert store["k_scalar"] == "x"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
