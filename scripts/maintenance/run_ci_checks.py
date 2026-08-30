"""
CI check runner — SINGLE SOURCE OF TRUTH for the test stage.

Both the local pre-push hook (.pre-commit-config.yaml) and the GitHub Actions
`test` job (ci.yml) invoke THIS script so that "local hook passes" => "CI test
passes" is guaranteed by construction. Every command here mirrors exactly what
the CI `test` job runs; the local hook therefore exercises a SUPERSET of what CI
does (it also runs ruff/mypy/bandit beforehand), so a green local hook cannot be
followed by a red CI test stage.

Coverage gate (--cov-fail-under=55) is enforced here, identical to CI.

Run:  python scripts/maintenance/run_ci_checks.py
Exit:  0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# Files the CI `test` job runs explicitly, kept as a list (NOT a directory glob)
# so we never silently start running integration tests that have not been
# verified to pass in CI. If a new file is proven green, add it here deliberately.
# Mirrors ci.yml "Run Core Integrity Tests".
INTEGRATION_FILES = [
    "tests/integration/test_rag_integration.py",
    "tests/integration/test_streaming_response.py",
    "tests/security/test_cache_security.py",
    "tests/integration/test_caching_system.py",
    "tests/integration/test_api_auth_login.py",
    "tests/integration/test_api_pdf_serving.py",
    "tests/integration/test_global_exception_handler.py",
    "tests/integration/test_ownership_hardening.py",
    "tests/integration/test_pdf_library_retention.py",
    "tests/integration/test_stream_error_isolation.py",
    "tests/integration/test_api_endpoints.py",
    # Ollama 비의존 UI 렌더러 + 세션 동시성 가드 (이전 CI 누락분).
    # test_chat_expander_verify_b.py: _render_build_progress_block의 유일한 가드.
    "tests/integration/test_chat_expander_verify_b.py",
    "tests/integration/test_thread_safety.py",
]

# Must stay identical to ci.yml coverage gate.
COVERAGE_FAIL_UNDER = 55


def _run(cmd: list[str], name: str) -> bool:
    print(f"\n[CI-CHECK] {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"[CI-CHECK] FAILED: {name}")
        return False
    print(f"[CI-CHECK] PASSED: {name}")
    return True


def run_integration() -> bool:
    ok = True
    for f in INTEGRATION_FILES:
        ok &= _run([sys.executable, "-m", "pytest", f], f"integration:{f}")
    return ok


def run_unit_with_coverage() -> bool:
    # Mirrors ci.yml: pytest --cov=src --cov-fail-under=55 tests/unit/
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=src",
        "--cov-report=term",
        "--cov-report=xml",
        "--disable-warnings",
        f"--cov-fail-under={COVERAGE_FAIL_UNDER}",
        "tests/unit/",
    ]
    return _run(cmd, "unit+coverage")


def run_scripts_import_sweep() -> bool:
    # scripts/과 tests/는 pytest 컬렉션·ruff 제외 범위라 리팩터링으로 인한
    # 끊긴 first-party 참조가 모든 기존 게이트에서 은닉된다. 이 스윕이 그 백스톱이다.
    cmd = [
        sys.executable,
        "scripts/maintenance/check_imports.py",
        "--paths",
        "scripts",
        "tests",
    ]
    return _run(cmd, "scripts-import-sweep")


def main() -> int:
    # Ensure src is importable (CI sets PYTHONPATH=workspace/src; mimic it).
    os.environ["PYTHONPATH"] = (
        str(ROOT_DIR / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

    passed = True
    passed &= run_scripts_import_sweep()
    passed &= run_integration()
    # Run unit coverage regardless, but only report overall failure if either failed.
    passed &= run_unit_with_coverage()

    if not passed:
        print("\n[CI-CHECK] One or more checks FAILED.")
        return 1
    print("\n[CI-CHECK] All checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
