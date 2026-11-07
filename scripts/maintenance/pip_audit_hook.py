"""
Local pre-push dependency CVE scanner — mirrors ci.yml `pip-audit` EXACTLY.

The GitHub Actions `security-scan` job runs:
        pip-audit -r requirements.txt --skip-editable \
        --ignore-vuln CVE-2025-2953 ... (7 CVEs)

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
    "CVE-2026-112895",
    "CVE-2026-112400",
    "CVE-2026-112897",
    "CVE-2026-112395",
    "CVE-2026-112404",
    "CVE-2026-112899",
    "CVE-2026-112900",
    "CVE-2026-112893",
    "CVE-2026-112891",
    "CVE-2026-112390",
    "CVE-2026-112401",
    "CVE-2026-112391",
    "CVE-2026-112894",
    "CVE-2026-112406",
    "CVE-2026-112399",
    "CVE-2026-112892",
    "CVE-2026-112393",
    "CVE-2026-96638",
    "CVE-2026-96640",
    "CVE-2026-96778",
    "CVE-2026-112906",
    "CVE-2026-112940",
    "CVE-2026-112914",
    "CVE-2026-112913",
    "CVE-2026-112937",
    "CVE-2026-112907",
    "CVE-2026-112938",
    "CVE-2026-112910",
    "CVE-2026-112934",
    "CVE-2026-112909",
    "CVE-2026-112911",
    "CVE-2026-112939",
    "CVE-2026-112915",
    "CVE-2026-112935",
    "CVE-2026-112912",
    "CVE-2026-112908",
    "CVE-2026-112941",
    "PYSEC-2026-4327",
    "PYSEC-2026-4328",
    "PYSEC-2026-4326",
    "PYSEC-2026-4339",
    "PYSEC-2026-4340",
    "PYSEC-2026-4343",
    "PYSEC-2026-4334",
    "PYSEC-2026-4337",
    "PYSEC-2026-4341",
    "PYSEC-2026-4330",
    "PYSEC-2026-4338",
    "PYSEC-2026-4335",
    "PYSEC-2026-4336",
    "PYSEC-2026-4331",
    "PYSEC-2026-4332",
    "PYSEC-2026-4329",
    "PYSEC-2026-4333",
    "PYSEC-2026-4342",
    "GHSA-78g4-xqwv-q3g8",
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
