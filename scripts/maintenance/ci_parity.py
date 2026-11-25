"""Shared gate runner for the local pre-push hook and GitHub Actions.

The same gate commands run locally and in CI. CI also runs setup, install, and
artifact upload steps that the local hook does not run. Branch protection is
what makes CI block merges. A green hook does not guarantee a green CI run.

The pre-push hook (.pre-commit-config.yaml) calls this script with
``--stage pre-push`` (all gating groups). CI jobs (.github/workflows/ci.yml)
call it with ``--only <group>`` (one group per job). Commands, arguments,
environment and scope come from this file so the two entry points do not drift.

The runner enforces hard timeouts per step and for the whole run, so a hanging
test or tool cannot wedge a push or a CI job. Timeouts kill the whole process
tree, which matters on Windows where child pytest workers survive a naive
``kill``.

Groups
------
static    ruff check . / ruff format --check . / mypy src
security  bandit -r src/ -ll / pip-audit -r requirements.txt --skip-editable
tests     scripts import sweep / gating integration / unit + coverage
extended  e2e + stability + perf (nightly, non-gating; see ci-extended.yml)

Run
---
    python scripts/maintenance/ci_parity.py --stage pre-push
    python scripts/maintenance/ci_parity.py --only static
    python scripts/maintenance/ci_parity.py --only tests
    python scripts/maintenance/ci_parity.py --list
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Keep this in sync with ci.yml. The pre-push hook enforces the same gate, so
# lowering it would silently weaken both.
COVERAGE_FAIL_UNDER = 70

# Whole-run hard cap. The measured pre-push run is ~3-4 minutes; the cap exists
# to fail loudly instead of hanging forever, not to constrain normal runs.
TOTAL_TIMEOUT_SECONDS = 900

# Environment parity with ci.yml's `test` job. Values are copied verbatim so a
# local run cannot silently diverge from CI (e.g. .env overrides, GPU probing,
# thread-count-dependent concurrency behavior).
CI_ENV: dict[str, str] = {
    "PYTHONPATH": str(ROOT_DIR / "src"),
    "IS_CI_TEST": "true",
    "TEST_API_KEY": "sk_admin_test_token_12345",
    "CUDA_VISIBLE_DEVICES": "",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    # conftest/config load .env via python-dotenv; CI has no .env file. Disable
    # it for parity runs so a developer's local .env cannot change test results.
    "DOTENV_DISABLE": "1",
}

CI_TOOLS_FILE = ROOT_DIR / "requirements" / "ci-tools.txt"

# Name -> (console script, module fallback). The version check uses _tool() so
# it inspects the same binary the gate will run.
VERSION_TOOLS: dict[str, tuple[str, str]] = {
    "ruff": ("ruff", "ruff"),
    "mypy": ("mypy", "mypy"),
    "bandit": ("bandit", "bandit"),
    "pip-audit": ("pip-audit", "pip_audit"),
}

OS_ENV_ALLOWLIST = (
    "PATH",  # executable lookup on both platforms
    "PATHEXT",  # executable extensions on Windows
    "COMSPEC",  # shell used by subprocess on Windows
    "SYSTEMROOT",  # Windows DLL and CRT lookup
    "WINDIR",  # Windows system directory lookup
    "SYSTEMDRIVE",  # Windows system drive lookup
    "TEMP",  # temporary files on Windows
    "TMP",  # temporary files on Windows
    "TMPDIR",  # temporary files on POSIX
    "HOME",  # user config location on POSIX
    "USERPROFILE",  # user profile location on Windows
    "HOMEDRIVE",  # home drive on Windows
    "HOMEPATH",  # home path on Windows
    "APPDATA",  # RoamingAppData on Windows
    "LOCALAPPDATA",  # LocalAppData on Windows
    "PROGRAMDATA",  # shared program data on Windows
    "NUMBER_OF_PROCESSORS",  # default worker count on Windows
    "LANG",  # locale and encoding on POSIX
    "LC_ALL",  # locale override on POSIX
)


def _gate_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in OS_ENV_ALLOWLIST if key in os.environ}
    env.update(CI_ENV)
    return env


# Paths that make CI (and therefore the pre-push hook) relevant. This is a
# SUPERSET of ci.yml's `on.*.paths`: pre-push may run when CI would not, but it
# must never skip when CI would run.
TRIGGER_PREFIXES = (
    "src/",
    "tests/",
    "scripts/",
    "requirements/",
    ".github/workflows/",
)
TRIGGER_FILES = (
    "pyproject.toml",
    "config.yml",
    ".pre-commit-config.yaml",
    "requirements.txt",
    "requirements-win.txt",
)

# Gating integration files. Each entry is deterministic and Ollama-free; the
# runner executes them in ONE pytest process to amortize the ~5-6s heavy import
# cost per process. Failures here are real regressions, not flaky infra.
GATING_INTEGRATION_FILES: tuple[str, ...] = (
    "tests/integration/test_rag_integration.py",
    "tests/integration/test_streaming_response.py",
    "tests/security/test_cache_security.py",
    "tests/security/test_security_suite.py",
    "tests/security/test_diskcache_pickle_rce_regression.py",
    "tests/integration/test_caching_system.py",
    "tests/integration/test_api_auth_login.py",
    "tests/integration/test_api_pdf_serving.py",
    "tests/integration/test_global_exception_handler.py",
    "tests/integration/test_ownership_hardening.py",
    "tests/integration/test_pdf_library_retention.py",
    "tests/integration/test_stream_error_isolation.py",
    "tests/integration/test_api_endpoints.py",
    "tests/integration/test_chat_expander_verify_b.py",
    "tests/integration/test_thread_safety.py",
    # Promoted deterministic coverage (previously excluded from CI).
    "tests/stability/test_session_concurrency.py",
    "tests/integration/test_pipeline_build_once.py",
    "tests/integration/test_rag_state_propagation.py",
    "tests/integration/test_session_management.py",
    "tests/integration/test_auth_bootstrap.py",
    "tests/integration/test_stream_ttft.py",
    "tests/integration/test_incremental_citation_dispatch.py",
    "tests/integration/test_metadata_propagation.py",
    "tests/integration/test_pipeline_edge_cases.py",
    "tests/integration/test_streamlit_ui_lifecycle.py",
    "tests/integration/test_ui_persistence_real.py",
    "tests/integration/test_dynamic_model_loading.py",
    "tests/integration/test_input_boundaries.py",
    "tests/integration/test_defects_verification.py",
    "tests/integration/test_rag_resource_manager_retirement.py",
)

# Real-model integration: needs a running Ollama + pulled models. Never gating;
# only the nightly ci-extended.yml `real-ollama` job runs this group, with
# IS_CI_TEST removed so the module-level skipif guards do not fire.
REAL_MODEL_FILES: tuple[str, ...] = (
    "tests/integration/test_upload_flow.py",
    "tests/integration/test_consecutive_queries.py",
    "tests/integration/test_resource_optimization.py",
)

# Non-gating (nightly) coverage: browser e2e, long perf, resource stability.
EXTENDED_PATHS: tuple[str, ...] = (
    "tests/stability",
    "tests/e2e",
    "tests/integration/test_performance_monitor.py",
)


@dataclass(frozen=True)
class Step:
    name: str
    group: str
    argv: tuple[str, ...]
    timeout: int
    stream: bool = False
    unset_env: tuple[str, ...] = ()


@dataclass
class StepResult:
    step: Step
    returncode: int
    duration: float
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _tool(binary: str, module: str) -> list[str]:
    """Prefer the console script (CI parity) but fall back to ``-m module``."""
    found = shutil.which(binary)
    if found:
        return [found]
    return [sys.executable, "-m", module]


_PIN_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^\s;#]+)",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)*)\b")


def _pinned_tool_versions() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in CI_TOOLS_FILE.read_text(encoding="utf-8").splitlines():
        match = _PIN_RE.match(line)
        if match:
            pins[match.group("name").lower()] = match.group("version")
    return pins


def _installed_tool_version(argv: list[str]) -> str:
    result = subprocess.run(
        [*argv, "--version"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "<unavailable>"
    for stream in (result.stdout, result.stderr):
        match = _VERSION_RE.search(stream)
        if match:
            return match.group(1)
    return "<unknown>"


def _preflight_tool_versions() -> bool:
    try:
        pins = _pinned_tool_versions()
    except OSError as exc:
        print(f"[ci-parity] FAILED: cannot read {CI_TOOLS_FILE}: {exc}", flush=True)
        return False

    problems: list[str] = []
    for name, (binary, module) in VERSION_TOOLS.items():
        expected = pins.get(name)
        if expected is None:
            problems.append(f"{name}: no pin in {CI_TOOLS_FILE.name}")
            continue
        actual = _installed_tool_version(_tool(binary, module))
        if actual != expected:
            problems.append(f"{name}: installed {actual}, expected {expected}")

    if not problems:
        return True
    print("[ci-parity] FAILED: gate tool versions do not match CI pins.", flush=True)
    for problem in problems:
        print(f"[ci-parity]   {problem}", flush=True)
    install_cmd = f"python -m pip install -r {CI_TOOLS_FILE.relative_to(ROOT_DIR)}"
    print(f"[ci-parity]   Install with: {install_cmd}", flush=True)
    return False


def _build_steps() -> list[Step]:
    py = sys.executable
    ruff = _tool("ruff", "ruff")
    mypy = _tool("mypy", "mypy")
    bandit = _tool("bandit", "bandit")
    pip_audit = _tool("pip-audit", "pip_audit")

    return [
        Step(
            name="ruff-check",
            group="static",
            argv=(*ruff, "check", "."),
            timeout=180,
        ),
        Step(
            name="ruff-format",
            group="static",
            argv=(*ruff, "format", "--check", "."),
            timeout=180,
        ),
        Step(
            name="mypy",
            group="static",
            argv=(*mypy, "src"),
            timeout=300,
        ),
        Step(
            name="bandit",
            group="security",
            argv=(*bandit, "-r", "src/", "-ll"),
            timeout=180,
        ),
        Step(
            name="pip-audit",
            group="security",
            argv=(*pip_audit, "-r", "requirements.txt", "--skip-editable"),
            timeout=240,
        ),
        Step(
            name="import-sweep",
            group="tests",
            argv=(
                py,
                "scripts/maintenance/check_imports.py",
                "--paths",
                "scripts",
                "tests",
            ),
            timeout=180,
        ),
        Step(
            name="integration",
            group="tests",
            argv=(py, "-m", "pytest", *GATING_INTEGRATION_FILES),
            timeout=600,
            stream=True,
        ),
        Step(
            name="unit+coverage",
            group="tests",
            argv=(
                py,
                "-m",
                "pytest",
                "--cov=src",
                "--cov-report=term",
                "--cov-report=xml",
                "--disable-warnings",
                f"--cov-fail-under={COVERAGE_FAIL_UNDER}",
                "tests/unit/",
            ),
            timeout=900,
            stream=True,
        ),
        Step(
            name="extended",
            group="extended",
            argv=(py, "-m", "pytest", *EXTENDED_PATHS),
            timeout=1800,
            stream=True,
        ),
        Step(
            name="real-model",
            group="real",
            argv=(py, "-m", "pytest", *REAL_MODEL_FILES),
            timeout=1800,
            stream=True,
            unset_env=("IS_CI_TEST",),
        ),
    ]


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _changed_files() -> list[str] | None:
    """Files changed by the push, or None when the range cannot be determined."""
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")
    if from_ref and to_ref and set(from_ref) != {"0"}:
        diff_range = f"{from_ref}..{to_ref}"
    else:
        upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if not upstream or not upstream.strip():
            return None
        diff_range = f"{upstream.strip()}..HEAD"
    out = _git("diff", "--name-only", diff_range)
    if out is None:
        return None
    return [line for line in out.splitlines() if line.strip()]


def _touches_ci_scope(files: list[str]) -> bool:
    for path in files:
        normalized = path.replace("\\", "/")
        if normalized in TRIGGER_FILES:
            return True
        if normalized.startswith(TRIGGER_PREFIXES):
            return True
    return False


def _local_changes() -> list[str]:
    git_queries = (
        ("diff", "--name-only"),
        ("diff", "--cached", "--name-only"),
        ("ls-files", "--others", "--exclude-standard"),
    )
    files: set[str] = set()
    for args in git_queries:
        output = _git(*args)
        if output:
            files.update(line for line in output.splitlines() if line.strip())
    return sorted(files)


def _blocking_local_changes() -> list[str]:
    return sorted(path for path in _local_changes() if _touches_ci_scope([path]))


def _abort_on_local_changes() -> bool:
    blocking = _blocking_local_changes()
    if not blocking:
        return False
    print(
        "[ci-parity] FAILED: pre-push runs against committed files. "
        "Commit or stash these gate-scope paths:",
        flush=True,
    )
    for path in blocking:
        print(f"[ci-parity]   {path}", flush=True)
    print(
        "[ci-parity]   Use --force-all to run gates against this worktree.", flush=True
    )
    return True


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:  # pragma: no cover - POSIX fallback
        proc.kill()


def _run_step(step: Step) -> StepResult:
    env = _gate_env()
    for key in step.unset_env:
        env.pop(key, None)
    started = time.perf_counter()
    print(f"[ci-parity] >>> {step.name}: {' '.join(step.argv)}", flush=True)

    if step.stream:
        proc = subprocess.Popen(
            list(step.argv),
            cwd=str(ROOT_DIR),
            env=env,
            stdout=None,
            stderr=subprocess.STDOUT,
        )
        timed_out = False
        try:
            proc.wait(timeout=step.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                pass
        duration = time.perf_counter() - started
        return StepResult(
            step,
            proc.returncode if proc.returncode is not None else -1,
            duration,
            "",
            timed_out,
        )

    proc = subprocess.Popen(
        list(step.argv),
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    timed_out = False
    try:
        raw, _ = proc.communicate(timeout=step.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            raw, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            raw = b""
    duration = time.perf_counter() - started
    output = (raw or b"").decode("utf-8", errors="replace")
    return StepResult(
        step,
        proc.returncode if proc.returncode is not None else -1,
        duration,
        output,
        timed_out,
    )


def _tail(text: str, limit: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


def _report(result: StepResult) -> None:
    status = "TIMEOUT" if result.timed_out else ("PASS" if result.ok else "FAIL")
    print(
        f"[ci-parity] {status:<7} {result.step.name:<16} {result.duration:6.1f}s",
        flush=True,
    )
    if not result.ok and result.output:
        print(f"----- {result.step.name} output (tail) -----", flush=True)
        print(_tail(result.output), flush=True)
        print(f"----- end {result.step.name} -----", flush=True)


def _run_group(steps: list[Step], parallel: bool) -> bool:
    if not steps:
        return True
    results: list[StepResult] = []
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(steps)) as pool:
            futures = {pool.submit(_run_step, step): step for step in steps}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        order = {step.name: i for i, step in enumerate(steps)}
        results.sort(key=lambda r: order[r.step.name])
    else:
        results = [_run_step(step) for step in steps]
    for result in results:
        _report(result)
    return all(result.ok for result in results)


def _select_steps(steps: list[Step], groups: list[str]) -> list[Step]:
    return [step for step in steps if step.group in groups]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["pre-push"],
        help="Run all gating groups (pre-push hook).",
    )
    parser.add_argument(
        "--only",
        choices=["static", "security", "tests", "extended", "real"],
        help="Run a single group (CI jobs).",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Ignore changed-file scoping and always run.",
    )
    parser.add_argument("--list", action="store_true", help="List steps and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    steps = _build_steps()

    if args.list:
        for step in steps:
            print(f"{step.group:<9} {step.name:<16} timeout={step.timeout}s")
        return 0

    if args.stage == "pre-push" and not args.force_all:
        if _abort_on_local_changes():
            return 1

    gating = args.stage == "pre-push" or args.only in {"static", "security", "tests"}
    if gating and not _preflight_tool_versions():
        return 1

    if args.only:
        selected = _select_steps(steps, [args.only])
        if args.only in {"extended", "real"}:
            return 0 if _run_group(selected, parallel=False) else 1
        parallel = args.only in {"static", "security"}
        return 0 if _run_group(selected, parallel=parallel) else 1

    # Pre-push stage: skip when the pushed commits cannot affect CI.
    if not args.force_all:
        changed = _changed_files()
        if changed is not None and not _touches_ci_scope(changed):
            print(
                "[ci-parity] No src/tests/scripts/requirements changes in this push; "
                "skipping gating checks (same as ci.yml path filters).",
                flush=True,
            )
            return 0

    started = time.perf_counter()
    static_security = _select_steps(steps, ["static", "security"])
    if not _run_group(static_security, parallel=True):
        print("[ci-parity] Static/security gates FAILED.", flush=True)
        return 1

    tests = _select_steps(steps, ["tests"])
    if not _run_group(tests, parallel=False):
        print("[ci-parity] Test gates FAILED.", flush=True)
        return 1

    elapsed = time.perf_counter() - started
    if elapsed > TOTAL_TIMEOUT_SECONDS:
        print(
            f"[ci-parity] FAILED: total runtime {elapsed:.1f}s exceeded "
            f"{TOTAL_TIMEOUT_SECONDS}s budget.",
            flush=True,
        )
        return 1

    print(f"[ci-parity] All gating checks PASSED in {elapsed:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
