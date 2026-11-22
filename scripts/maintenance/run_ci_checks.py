"""Deprecated alias for ``scripts/maintenance/ci_parity.py --only tests``.

Kept so existing docs, CI snippets and muscle memory keep working. The gating
commands now live in ci_parity.py (single source of truth shared with ci.yml and
the pre-push hook).

Run:  python scripts/maintenance/run_ci_checks.py
Exit:  0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_parity import main as _ci_parity_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return _ci_parity_main(["--only", "tests", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    sys.exit(main())
