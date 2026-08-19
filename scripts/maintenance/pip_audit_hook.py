"""
Local pre-push dependency CVE scanner — mirrors ci.yml `pip-audit` EXACTLY.

The GitHub Actions `security-scan` job runs:
    pip-audit -r requirements.txt --skip-editable \
        --ignore-vuln CVE-2025-2953 ... (8 CVEs)

This script reproduces the same invocation so the local pre-push hook fails on
the SAME dependency vulns CI would, keeping the repo clean (no vuln-only CI
failures after a green hook). The ignore list MUST stay in sync with ci.yml.

Run:  python scripts/maintenance/pip_audit_hook.py
Exit:  0 if clean (or only ignored CVEs), 1 on real findings.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

REQUIREMENTS = ROOT_DIR / "requirements.txt"

# Keep in sync with .github/workflows/ci.yml `pip-audit --ignore-vuln` list.
IGNORED_CVES = [
    "CVE-2025-2953",
    "CVE-2025-3730",
    "CVE-2025-67221",
    "CVE-2026-26013",
    "CVE-2026-27794",
    "CVE-2026-28277",
    "CVE-2026-28500",
]


def main() -> int:
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
    cmd += [f"--ignore-vuln={cve}" for cve in IGNORED_CVES]

    print(f"[pip-audit] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
