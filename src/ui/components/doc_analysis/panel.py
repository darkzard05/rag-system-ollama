"""Single st.status renderer for the document analysis panel (1A).

Reads ``BuildState`` only — flat legacy keys are never consulted (3B).
No polling timer: refresh arrives via worker state-propagation reruns,
plus error surfacing on worker failure (D4, ``async_worker._on_complete``).
"""

from __future__ import annotations

import streamlit as st

from core.build_state import BuildState
from core.session import SessionManager
from ui.components.common import ui_error
from ui.components.doc_analysis.actions import on_cancel, on_retry
from ui.components.doc_analysis.viewmodel import _RUNNING_PHASES, to_viewmodel
from ui.strings import t
from ui.widget_keys import cancel_rebuild_key, retry_build_key


def _doc_cache_override(sid: str, build: BuildState) -> bool | None:
    """Prefer fresh doc_stats cache flag; fall back to BuildState value."""
    # A resolved BuildState value is authoritative: skip the session lookup
    # (and its lock acquisition) on every render frame.
    if build.cache_used is not None:
        return build.cache_used
    doc_stats = SessionManager.get("doc_stats", {}, sid)
    if isinstance(doc_stats, dict) and "cache_used" in doc_stats:
        value = doc_stats.get("cache_used")
        if isinstance(value, bool):
            return value
    return build.cache_used


def render_document_panel(sid: str) -> None:
    """Render the document analysis card (single st.status call)."""
    build = SessionManager.get_build_state(session_id=sid)
    if build is None:
        return
    vm = to_viewmodel(build, cache_used=_doc_cache_override(sid, build))
    if vm is None:
        return
    try:
        attempt = int(build.attempt)
    except (TypeError, ValueError):
        attempt = 0

    with st.status(
        vm.label,
        expanded=vm.expanded,
        state=vm.status_state,  # type: ignore[arg-type]
    ) as status_container:
        if vm.progress is not None:
            st.progress(vm.progress)
        if vm.stepper:
            st.caption(vm.stepper)
        if vm.caption:
            st.caption(vm.caption)
        if vm.show_error:
            error = (
                build.error or SessionManager.get("pdf_processing_error", "", sid) or ""
            )
            if error:
                ui_error(str(error))
        if "cancel" in vm.actions:
            # Once cancellation is requested, hide the CTA and let the
            # post-block update below settle the final visual state.
            if not bool(SessionManager.get("rebuild_cancelled", False, sid)):
                st.button(
                    t("action_cancel_build"),
                    key=cancel_rebuild_key(sid, attempt),
                    on_click=on_cancel,
                    args=(sid,),
                    width="stretch",
                )
        elif "retry" in vm.actions:
            # CANCELLED shows the retry CTA directly (2B); ERROR shows it
            # below the friendly error message.
            retry_label = t("action_retry_build")
            st.button(
                retry_label,
                key=retry_build_key(sid, attempt),
                on_click=on_retry,
                args=(sid,),
                help=t("action_retry_build_help"),
                width="stretch",
            )

    # st.status.__exit__ force-transitions a running container to
    # state="complete" + expanded=False on block exit, which strobes the
    # panel and swallows the cancel click. Re-assert the viewmodel state
    # only after __exit__ has run so running/error panels stay expanded.
    status_container.update(
        label=vm.label,
        expanded=vm.expanded,
        state=vm.status_state,  # type: ignore[arg-type]
    )


def should_mount_fragment(sid: str) -> bool:
    """Fragment mounts only while the build is actively running (4A, no timer).

    Terminal phases (DONE/ERROR/CANCELLED) and IDLE render via the static
    block path — mounting a fragment there would idle-poll (fix-001).
    """
    build = SessionManager.get_build_state(session_id=sid)
    return build is not None and build.phase in _RUNNING_PHASES
