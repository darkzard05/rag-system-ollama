"""Bounded explicit-state verification of the chat auto-scroll latch.

Property (plain words):
    While a response streams, a reader who is pinned to the bottom is carried
    to every new bottom; a reader who scrolled up to read history is never
    yanked down. The only JavaScript write of ``scrollTop`` is
    ``scrollToBottom``'s rAF callback, and the ``scroll`` listener must not
    mistake the scroll that callback just made for user intent.

Why an explicit-state explorer:
    The failure is an ordering bug (wheel vs. delayed ``scroll`` dispatch vs.
    rAF vs. rerun), not a value bug. TLA+/Java and Z3 are not installed in
    this environment, so this checker enumerates bounded interleavings with
    the repository's own Python 3.11 and no third-party dependencies.

Layers:
    A. Source tether - anchors the model to the real implementation. If the
       JS/CSS no longer contains the rules the model assumes, the check
       fails before any modelling claim is made.
    B. Model - enumerates every interleaving of the latch events up to
       --depth and checks the safety properties below.

Events map to code in ``src/ui/ui.py::inject_custom_css`` (JS string):
    grow            DOM append that triggers the MutationObserver (line 289)
    observer        MutationObserver callback -> scrollToBottom(false) (290)
    raf             requestAnimationFrame callback (188-199)
    wheel_up /      wheel handler (231-257)
    wheel_down
    wheel_landing   80 ms post-layout remeasure (247-253)
    user_scroll     browser-driven scrollTop change (scrollbar/keys/touch);
                    this is NOT a JS write, and it schedules a scroll event
    scroll_dispatch scroll handler (266-285)
    submit          Enter/send-button hook (304-316)
    force_timer     the 50 ms setTimeout after submit (307, 314)
    rerun           observer disconnect/reinstall + initial-yank gate (155-159,
                    323-328)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
UI_PY = ROOT / "src" / "ui" / "ui.py"
MAIN_CSS = ROOT / "src" / "ui" / "styles" / "main.css"
SENTINEL_PY = ROOT / "src" / "ui" / "components" / "chat_autoscroll.py"

STICK = 8  # BOTTOM_STICK_PX, imported at src/ui/ui.py:120
CLIENT_H = 400  # fixed viewport for the bounded model
START_H = 1000
MAX_H = 1080
GROW_STEPS = (1, 40)
USER_SCROLL_STEPS = (-200, -40, 40)
# Queued events of the same kind are observationally equivalent beyond two:
# the latch only consults the current geometry, not the queue length.
MAX_QUEUED = 2
# Force-budget counters saturate: exceeding 1 + submits is already a finding.
MAX_COUNT = 3


class TetherFailure(Exception):
    """The source no longer matches the rules the model encodes."""


@dataclass(frozen=True)
class Faults:
    """Model mutations used only by --self-test to prove checks can fail."""

    no_raf_recheck: bool = False
    no_own_yank_guard: bool = False
    force_on_every_rerun: bool = False


@dataclass(frozen=True)
class State:
    up: bool
    h: int
    t: int
    ticking: bool
    raf_force: bool
    yank_target: int
    initial_yanked: bool
    forced_yanks: int
    submits: int
    wheel_resume: bool
    pending_scrolls: int
    pending_observer: int
    pending_forces: int
    last_scroll_top: int


# State right after the first setupChatObserver() run at page load:
# initialYanked was false, so the code set it and called scrollToBottom(true).
# scrollTop = scrollHeight is clamped to h - clientHeight, the browser queues
# one scroll event for that write, and __lastScrollTop still holds the value
# captured before the yank (normally 0 on a fresh load).
INITIAL = State(
    up=False,
    h=START_H,
    t=START_H - CLIENT_H,
    ticking=False,
    raf_force=False,
    yank_target=START_H,
    initial_yanked=True,
    forced_yanks=1,
    submits=0,
    wheel_resume=False,
    pending_scrolls=1,
    pending_observer=0,
    pending_forces=0,
    last_scroll_top=0,
)

Event = tuple[str, ...]


def extract_js(source: str) -> str:
    """Return the inline autoscroll JS string from inject_custom_css."""
    marker = 'js = f"""'
    start = source.find(marker)
    if start < 0:
        raise TetherFailure("src/ui/ui.py: inline JS assignment not found")
    body_start = start + len(marker)
    end = source.find('"""', body_start)
    if end < 0:
        raise TetherFailure("src/ui/ui.py: inline JS string not terminated")
    return source[body_start:end]


def source_tether() -> list[str]:
    """Anchor every rule the model relies on to the real source text."""
    ui = UI_PY.read_text(encoding="utf-8")
    js = extract_js(ui)
    css = MAIN_CSS.read_text(encoding="utf-8")
    sentinel = SENTINEL_PY.read_text(encoding="utf-8")
    failures: list[str] = []

    def require(name: str, pattern: str, text: str, count: int = 1) -> None:
        found = len(re.findall(pattern, text))
        if found != count:
            failures.append(
                f"{name}: expected {count} match(es) in source, found {found}"
            )

    writers = re.findall(r"scrollTop\s*=", js)
    if len(writers) != 1:
        failures.append(
            "single_writer: expected exactly 1 JS `scrollTop =` write, "
            f"found {len(writers)}: {writers}"
        )

    require(
        "own_yank_guard",
        r"const yank = scrollState\.yankTarget;\s*"
        r"if \(yank >= 0 && container\.scrollTop >= yank - "
        r"container\.clientHeight - \{BOTTOM_STICK_PX\}\)",
        js,
    )
    require(
        "raf_recheck",
        r"if \(!force && scrollState\.userScrolledUp\)",
        js,
        count=2,
    )
    require(
        "yank_pair",
        r"scrollState\.yankTarget = container\.scrollHeight;\s*"
        r"container\.scrollTop = scrollState\.yankTarget;",
        js,
    )
    require(
        "wheel_up_latch",
        r"if \(e\.deltaY < 0\)[\s\S]{0,160}?"
        r"scrollState\.userScrolledUp = true;",
        js,
    )
    require(
        "wheel_down_release",
        r"if \(dist <= \{BOTTOM_STICK_PX\}\) \{\{\s*"
        r"scrollState\.userScrolledUp = false;",
        js,
    )
    require(
        "post_wheel_remeasure",
        r"__wheelResumePending = true;[\s\S]{0,600}?"
        r"landed <= \{BOTTOM_STICK_PX\}[\s\S]{0,120}?"
        r"scrollState\.userScrolledUp = false;",
        js,
    )
    require(
        "mutation_observer_follow",
        r"new MutationObserver\(\(\) => \{\{\s*scrollToBottom\(false\);",
        js,
    )
    require(
        "submit_force",
        r"scrollState\.userScrolledUp = false;\s*"
        r"setTimeout\(\(\) => scrollToBottom\(true\), 50\);",
        js,
        count=2,
    )
    require(
        "initial_yank_once",
        r"if \(!scrollState\.initialYanked\) \{\{\s*"
        r"scrollState\.initialYanked = true;\s*"
        r"scrollToBottom\(true\);\s*\}\} else \{\{\s*"
        r"scrollToBottom\(false\);\s*\}\}",
        js,
    )
    if not re.search(r"overflow-anchor:\s*none\s*;", css):
        failures.append("css_anchor: `overflow-anchor: none` missing from main.css")
    if re.search(r"overflow-anchor:\s*none\s*!important", css):
        failures.append(
            "css_anchor: `!important` must not be added to `overflow-anchor: none`"
        )
    if re.search(r"scrollTop\s*=", sentinel):
        failures.append(
            "observe_only_sentinel: chat_autoscroll.py must not write scrollTop"
        )
    return failures


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def schedule(state: State, force: bool) -> State:
    """Mirror scrollToBottom(): latch guard, then the isTicking gate."""
    if not force and state.up:
        return state
    if state.ticking:
        return state
    return replace(state, ticking=True, raf_force=force)


def enabled_events(state: State) -> list[Event]:
    events: list[Event] = []
    for delta in GROW_STEPS:
        if state.h + delta <= MAX_H:
            events.append(("grow", str(delta)))
    if state.pending_observer:
        events.append(("observer",))
    if state.ticking:
        events.append(("raf",))
    events.extend((("wheel_up",), ("wheel_down",)))
    if state.wheel_resume:
        events.append(("wheel_landing",))
    for delta in USER_SCROLL_STEPS:
        events.append(("user_scroll", str(delta)))
    if state.pending_scrolls:
        events.append(("scroll_dispatch",))
    events.append(("submit",))
    if state.pending_forces:
        events.append(("force_timer",))
    events.append(("rerun",))
    return events


def apply_event(state: State, event: Event, faults: Faults) -> tuple[State, list[str]]:
    """Apply one indivisible code step; return violations raised by it."""
    kind = event[0]
    violations: list[str] = []

    if kind == "grow":
        delta = int(event[1])
        return (
            replace(
                state,
                h=state.h + delta,
                pending_observer=min(MAX_QUEUED, state.pending_observer + 1),
            ),
            violations,
        )

    if kind == "observer":
        return (
            replace(
                schedule(state, force=False),
                pending_observer=state.pending_observer - 1,
            ),
            violations,
        )

    if kind == "raf":
        if not state.ticking:
            return state, violations
        if not state.raf_force and state.up and not faults.no_raf_recheck:
            # The rAF callback re-checks the latch before writing.
            return replace(state, ticking=False), violations
        if not state.raf_force and state.up:
            violations.append(
                "NO_WRITE_WHILE_USER_UP: rAF yanked while userScrolledUp=true"
            )
        # scrollTop = scrollHeight is clamped by the browser to h - clientHeight.
        next_t = max(0, state.h - CLIENT_H)
        # Assigning scrollTop makes the browser emit a scroll event; that
        # delayed event is exactly the one the own-yank guard must absorb.
        return (
            replace(
                state,
                t=next_t,
                yank_target=state.h,
                ticking=False,
                forced_yanks=min(MAX_COUNT, state.forced_yanks + int(state.raf_force)),
                pending_scrolls=min(MAX_QUEUED, state.pending_scrolls + 1),
            ),
            violations,
        )

    if kind == "wheel_up":
        return replace(state, up=True), violations

    if kind == "wheel_down":
        dist = state.h - state.t - CLIENT_H
        if dist <= STICK:
            return replace(state, up=False), violations
        if not state.wheel_resume:
            return replace(state, wheel_resume=True), violations
        return state, violations

    if kind == "wheel_landing":
        if not state.wheel_resume:
            return state, violations
        landed = state.h - state.t - CLIENT_H
        return (
            replace(
                state,
                wheel_resume=False,
                up=False if landed <= STICK else state.up,
            ),
            violations,
        )

    if kind == "user_scroll":
        delta = int(event[1])
        next_t = clamp(state.t + delta, 0, max(0, state.h - CLIENT_H))
        if next_t == state.t:
            return state, violations
        return (
            replace(
                state,
                t=next_t,
                pending_scrolls=min(MAX_QUEUED, state.pending_scrolls + 1),
            ),
            violations,
        )

    if kind == "scroll_dispatch":
        consumed = replace(state, pending_scrolls=state.pending_scrolls - 1)
        if state.last_scroll_top == state.t:
            # __lastScrollTop dedupe at line 267.
            return consumed, violations
        guarded = replace(consumed, last_scroll_top=state.t)
        own_yank = state.yank_target >= 0 and (
            state.t >= state.yank_target - CLIENT_H - STICK
        )
        if own_yank and not faults.no_own_yank_guard:
            # Delayed event caused by our own yank; leave the latch alone.
            return guarded, violations
        dist = state.h - state.t - CLIENT_H
        next_up = dist > STICK
        if own_yank and next_up != state.up:
            violations.append(
                "OWN_YANK_DOES_NOT_LATCH: delayed scroll event after yank "
                f"flipped userScrolledUp {state.up} -> {next_up} "
                f"(phantom dist={dist})"
            )
        return (
            replace(guarded, yank_target=-1, up=next_up),
            violations,
        )

    if kind == "submit":
        return (
            replace(
                state,
                up=False,
                submits=min(MAX_COUNT, state.submits + 1),
                pending_forces=min(MAX_QUEUED, state.pending_forces + 1),
            ),
            violations,
        )

    if kind == "force_timer":
        if not state.pending_forces:
            return state, violations
        return (
            schedule(
                replace(state, pending_forces=state.pending_forces - 1), force=True
            ),
            violations,
        )

    if kind == "rerun":
        if faults.force_on_every_rerun:
            after = schedule(state, force=True)
        elif not state.initial_yanked:
            after = schedule(replace(state, initial_yanked=True), force=True)
        else:
            after = schedule(state, force=False)
        after = replace(
            after,
            pending_observer=0,
            last_scroll_top=state.t,
        )
        # Rerun re-installs the observer with scrollToBottom(false)/force only;
        # a live latch must never be overridden by a non-force call.
        if state.up and state.initial_yanked and after.t != state.t:
            violations.append(
                "RERUN_RESPECTS_LATCH: rerun moved scrollTop while userScrolledUp=true"
            )
        return after, violations

    raise ValueError(f"unknown event: {kind!r}")


def state_violations(state: State) -> list[str]:
    violations: list[str] = []
    quiescent = (
        not state.up
        and not state.ticking
        and state.pending_scrolls == 0
        and state.pending_observer == 0
        and state.pending_forces == 0
        and not state.wheel_resume
    )
    if quiescent and state.h - state.t - CLIENT_H > STICK:
        violations.append(
            "BOTTOM_STICK: quiescent reader is not at the bottom "
            f"(h={state.h}, t={state.t}, dist={state.h - state.t - CLIENT_H})"
        )
    if state.forced_yanks > 1 + state.submits:
        violations.append(
            "FORCE_BUDGET: forced yanks exceed one initial yank plus "
            f"explicit submits ({state.forced_yanks} > {1 + state.submits})"
        )
    return violations


def trace(
    parents: dict[State, tuple[State | None, Event | None]], state: State
) -> list[str]:
    steps: list[str] = []
    current: State | None = state
    while current is not None:
        parent, event = parents[current]
        if event is not None:
            steps.append(" ".join(event))
        current = parent
    return list(reversed(steps))


def check(
    depth: int, faults: Faults, max_states: int = 200_000
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parents: dict[State, tuple[State | None, Event | None]] = {INITIAL: (None, None)}
    frontier: dict[State, int] = {INITIAL: 0}
    findings: list[dict[str, Any]] = []
    seen_kinds: set[str] = set()
    truncated = False

    def record(violation: str, state: State) -> None:
        kind = violation.split(":")[0]
        if kind in seen_kinds:
            return
        seen_kinds.add(kind)
        findings.append(
            {
                "kind": kind,
                "detail": violation,
                "steps": trace(parents, state),
            }
        )

    for violation in state_violations(INITIAL):
        record(violation, INITIAL)

    while frontier:
        if len(parents) >= max_states:
            truncated = True
            break
        next_frontier: dict[State, int] = {}
        for state, level in frontier.items():
            if level >= depth:
                continue
            for event in enabled_events(state):
                child, raised = apply_event(state, event, faults)
                if child not in parents:
                    parents[child] = (state, event)
                    next_frontier[child] = level + 1
                for violation in raised + state_violations(child):
                    record(violation, child)
        frontier = next_frontier

    summary = {
        "states_explored": len(parents),
        "depth": depth,
        "truncated": truncated,
        "h_bounds": [START_H, MAX_H],
        "client_height": CLIENT_H,
        "stick_px": STICK,
        "max_queued": MAX_QUEUED,
        "event_kinds": sorted({e[0] for _, e in parents.values() if e}),
        "faults": {
            "no_raf_recheck": faults.no_raf_recheck,
            "no_own_yank_guard": faults.no_own_yank_guard,
            "force_on_every_rerun": faults.force_on_every_rerun,
        },
    }
    return summary, findings


def self_test(depth: int, max_states: int) -> int:
    cases = (
        ("no_raf_recheck", Faults(no_raf_recheck=True), "NO_WRITE_WHILE_USER_UP"),
        (
            "no_own_yank_guard",
            Faults(no_own_yank_guard=True),
            "OWN_YANK_DOES_NOT_LATCH",
        ),
        ("force_on_every_rerun", Faults(force_on_every_rerun=True), "FORCE_BUDGET"),
    )
    failed = 0
    for name, faults, expected in cases:
        _, findings = check(depth, faults, max_states)
        kinds = {f["kind"] for f in findings}
        if expected in kinds:
            example = next(f for f in findings if f["kind"] == expected)
            print(f"  PASS {name}: caught {expected}")
            print(f"       trace: {' -> '.join(example['steps'])}")
        else:
            failed += 1
            print(f"  FAIL {name}: expected {expected}, got {sorted(kinds)}")
    return failed


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-states", type=int, default=200_000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", type=Path, help="write a JSON evidence report")
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("Layer A: source tether")
    tether_failures = source_tether()
    if tether_failures:
        for failure in tether_failures:
            print(f"  FAIL {failure}")
        print("Model results are not meaningful until the tether matches the code.")
        return 1
    print("  PASS: all anchored rules found in ui.py / main.css / chat_autoscroll.py")

    print("Layer B: bounded interleaving check")
    summary, findings = check(args.depth, Faults(), args.max_states)
    print(
        f"  explored {summary['states_explored']} states at depth {args.depth}"
        + (" (TRUNCATED)" if summary["truncated"] else "")
    )
    if findings:
        for finding in findings:
            print(f"  FAIL {finding['kind']}: {finding.get('detail', '')}")
            if finding["steps"]:
                print(f"       trace: {' -> '.join(finding['steps'])}")
    else:
        print("  PASS: no safety violation in the bounded model")

    self_test_failures = 0
    if args.self_test:
        print("Layer C: fault injection (checks must be able to fail)")
        self_test_failures = self_test(args.depth, args.max_states)

    if args.json:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sync_commit": "02c6258db1867999457a8523b1b4cd5407954f2d",
            "layer_a_failures": tether_failures,
            "layer_b": summary,
            "findings": findings,
            "self_test_failures": self_test_failures,
            "passed": (
                not tether_failures
                and not findings
                and not self_test_failures
                and not summary["truncated"]
            ),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  evidence: {args.json}")

    return int(
        bool(tether_failures)
        or bool(findings)
        or bool(self_test_failures)
        or bool(summary["truncated"])
    )


if __name__ == "__main__":
    sys.exit(main())
