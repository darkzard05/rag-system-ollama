"""
Local pre-push dependency CVE scanner — mirrors ci.yml `pip-audit` EXACTLY.

The GitHub Actions `security-scan` job runs:

        pip-audit -r requirements.txt --skip-editable

This script reproduces the same invocation so the local pre-push hook fails on
the SAME dependency vulns CI would, keeping the repo clean (no vuln-only CI
failures after a green hook).

There are intentionally NO `--ignore-vuln` exceptions: vulnerable packages are
pinned to their fixed versions in requirements/. If an advisory has no released
fix yet, the gate fails and the finding is tracked rather than suppressed.

Run:  python scripts/maintenance/pip_audit_hook.py
Exit:  0 if clean, 1 on real findings.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

REQUIREMENTS = ROOT_DIR / "requirements.txt"

# The fixed versions (pillow 13 / filelock 4.2) ship no Python <3.11 wheels, so
# auditing under an older interpreter cannot resolve the requirement set.
MIN_PYTHON = (3, 11)


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        print(
            "[pip-audit] Python >=3.11 required: pillow 13 / filelock 4.2 "
            f"(the security fixes) have no <3.11 wheels. Current interpreter: "
            f"{sys.version.split()[0]}. Recreate the environment on Python 3.11+.",
            file=sys.stderr,
        )
        return 1

    if not REQUIREMENTS.exists():
        print(f"[pip-audit] {REQUIREMENTS} not found, skipping.")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        str(REQUIREMENTS),
        "--skip-editable",
    ]

    print(f"[pip-audit] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
