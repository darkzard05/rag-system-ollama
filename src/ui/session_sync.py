# src/ui/session_sync.py
"""
UI-sync adapter that mirrors ``SessionManager`` state into Streamlit.

This module lives in ``src/ui`` (not ``src/core``) so the core session layer
stays free of any Streamlit import. ``SessionManager`` calls ``write``/``read``
on whatever adapter is installed via ``SessionManager.set_ui_sync``; when none
is installed it falls back to the global store only (see core.session.manager).

The ``write`` semantics intentionally reproduce the previous in-core behavior
(manager.py lines ~240-245): lists are sliced and dicts are ``copy.copy``'d
before assignment so the UI never aliases the internal mutable state.
"""

from __future__ import annotations

import copy
from typing import Any

import streamlit as st


class StreamlitSessionSync:
    """Pluggable adapter routing SessionManager state to ``st.session_state``."""

    @staticmethod
    def write(key: str, val: Any) -> None:
        """Mirror ``val`` into ``st.session_state[key]``.

        Lists are copied via slice and dicts via ``copy.copy`` to preserve the
        no-alias contract that the previous in-core sync relied on.
        """
        if isinstance(val, list):
            st.session_state[key] = val[:]
        elif isinstance(val, dict):
            st.session_state[key] = copy.copy(val)
        else:
            st.session_state[key] = val

    @staticmethod
    def read(key: str, default: Any = None) -> Any:
        """Read ``key`` from ``st.session_state`` falling back to ``default``."""
        return st.session_state.get(key, default)
