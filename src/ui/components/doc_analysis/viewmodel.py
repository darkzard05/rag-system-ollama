"""Pure BuildState -> BuildVM conversion (no Streamlit import).

Single source of truth for the document analysis panel. All branching
(running / done / cancelled / error / idle) lives here so it is unit
testable without Streamlit mocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.build_state import (
    PHASE_DETAIL_KEY,
    BuildPhase,
    BuildState,
    elapsed_sec,
    is_terminal,
)
from ui.components.common import status_line
from ui.strings import t


@dataclass(frozen=True)
class BuildVM:
    """Render-ready snapshot for the document analysis panel."""

    label: str
    status_state: str  # "running" | "complete" | "error"
    expanded: bool
    progress: float | None  # None omits the gauge (DONE)
    stepper: str | None  # running phases only; terminal -> None
    caption: str | None
    actions: tuple[str, ...]  # () | ("cancel",) | ("retry",)
    show_error: bool = False


_RUNNING_PHASES: frozenset[BuildPhase] = frozenset(
    {
        BuildPhase.PREPARING,
        BuildPhase.EXTRACTING,
        BuildPhase.CHUNKING,
        BuildPhase.INDEXING,
        BuildPhase.REGISTERING,
    }
)

_STEP_ORDER: tuple[BuildPhase, ...] = (
    BuildPhase.EXTRACTING,
    BuildPhase.CHUNKING,
    BuildPhase.INDEXING,
    BuildPhase.REGISTERING,
)


def _step_labels() -> dict[BuildPhase, str]:
    """Step labels resolved for the active locale (literal t() calls for the i18n AST gate)."""
    return {
        BuildPhase.EXTRACTING: t("build_step_extract"),
        BuildPhase.CHUNKING: t("build_step_chunk"),
        BuildPhase.INDEXING: t("build_step_index"),
        BuildPhase.REGISTERING: t("build_step_finalize"),
    }


def _i18n_anchor() -> tuple[str, ...]:
    """Keep dynamically-resolved detail keys visible to the i18n AST gate.

    ``to_viewmodel`` resolves the running phase text via
    ``PHASE_DETAIL_KEY`` (variable key), so the literal gate cannot see
    ``build_in_progress``/``build_analyzing``. Referencing them here as
    literals keeps ``gate_en_keys_read`` green without changing runtime.
    """
    return (t("build_in_progress"), t("build_analyzing"))


def stepper_caption(phase: BuildPhase) -> str | None:
    """One-line stepper: done phases get check, active gets triangle.

    Terminal phases (DONE/ERROR/CANCELLED) and IDLE return None — the
    panel does not render a stepper then (2B simplification; fixes the
    old bug where CANCELLED/ERROR fell through to ``current=0`` and
    wrongly highlighted EXTRACTING).
    """
    if phase not in _RUNNING_PHASES:
        return None
    labels = _step_labels()
    try:
        current = _STEP_ORDER.index(phase)
    except ValueError:
        # PREPARING is before the first tracked step: all pending.
        current = -1
    parts: list[str] = []
    for idx, step_phase in enumerate(_STEP_ORDER):
        label = labels[step_phase]
        if idx < current:
            marker = "✓"
        elif idx == current:
            marker = "▸"
        else:
            marker = "○"
        parts.append(f"{marker} {label}")
    # PREPARING shows all-pending (no active step yet).
    if phase == BuildPhase.PREPARING:
        parts = [f"○ {label}" for label in labels.values()]
    return status_line(*parts)


def to_viewmodel(
    build: BuildState | None, cache_used: bool | None = None
) -> BuildVM | None:
    """Convert a BuildState into a render-ready BuildVM.

    Returns None when nothing should render (IDLE without a file).
    ``cache_used`` overrides ``build.cache_used`` when the caller holds a
    fresher ``doc_stats`` value; None falls back to the state object.
    """
    if build is None:
        return None
    if build.phase == BuildPhase.IDLE and not build.file_name:
        return None
    file_name = build.file_name or t("doc_default_name")
    elapsed = elapsed_sec(build)
    tag = ""
    resolved_cache = cache_used if cache_used is not None else build.cache_used
    if resolved_cache is True:
        tag = t("cache_tag_reused")
    elif resolved_cache is False:
        tag = t("cache_tag_new")

    if build.phase in _RUNNING_PHASES:
        phase_text = t(PHASE_DETAIL_KEY.get(build.phase, "build_in_progress"))
        return BuildVM(
            label=f"{file_name} · {phase_text}",
            status_state="running",
            expanded=True,
            progress=max(0, min(100, build.pct)) / 100,
            stepper=stepper_caption(build.phase),
            caption=t("build_progress_caption").format(
                progress=build.pct, elapsed=elapsed
            ),
            actions=("cancel",),
        )

    if build.phase == BuildPhase.DONE:
        return BuildVM(
            label=f"{file_name} · 준비 완료{tag}",
            status_state="complete",
            expanded=False,
            progress=None,
            stepper=None,
            caption=(
                t("build_complete_caption").format(elapsed=elapsed)
                if elapsed > 0
                else t("build_complete_default")
            ),
            actions=(),
        )

    if build.phase == BuildPhase.CANCELLED:
        # 2B: cancelled is terminal — error chrome + retry, no spinner/stepper.
        return BuildVM(
            label=f"{file_name} · {t('build_cancelled').replace('...', '').strip()}",
            status_state="error",
            expanded=True,
            progress=None,
            stepper=None,
            caption=t("build_progress_caption").format(
                progress=build.pct, elapsed=elapsed
            ),
            actions=("retry",),
        )

    # ERROR (includes any unknown terminal fallback via is_terminal guard).
    if build.phase == BuildPhase.ERROR or is_terminal(build.phase):
        return BuildVM(
            label=f"{file_name} · {t('build_phase_error')}",
            status_state="error",
            expanded=True,
            progress=None,
            stepper=None,
            caption=t("build_progress_caption").format(
                progress=build.pct, elapsed=elapsed
            ),
            actions=("retry",),
            show_error=True,
        )

    return None
