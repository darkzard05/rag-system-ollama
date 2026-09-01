"""
Import-sweep guard for ad-hoc scripts.

Catches stale first-party references in ``scripts/`` (e.g. ``from X import Y``
where ``Y`` was renamed/removed by a refactor). pytest never collects
non-``test_*`` scripts and ruff excludes ``scripts/``/``tests/``, so such rot is
invisible to every existing CI gate — this sweep is the backstop.

Resolution mirrors the audit methodology: parse AST, then resolve each
first-party ``from X import Y`` / ``import X.Y`` against the live ``src``
package (``importlib`` + ``hasattr``). A missing module/name is reported and
fails the run. Submodule imports (``from ui.components import chat``) resolve
correctly because ``hasattr`` on the parent package triggers submodule loading.

Run (from repo root):
    python scripts/maintenance/check_imports.py [--paths scripts tests]
Exit: 0 if every first-party reference resolves, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import sys
from pathlib import Path

FIRST_PARTY_PREFIXES = (
    "core",
    "common",
    "cache",
    "ui",
    "api",
    "security",
    "infra",
    "services",
    "src.",
)

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = ROOT_DIR / "src"


def _is_first_party(module: str) -> bool:
    return any(module.startswith(p) for p in FIRST_PARTY_PREFIXES)


def _resolve_module(module: str) -> tuple[object | None, Exception | None]:
    """Import a first-party module, stripping the optional ``src.`` prefix.

    Importing ``src.core.x`` registers the same module objects as ``core.x``
    (the conftest alias logic), so resolving either way is equivalent.
    """
    if module.startswith("src."):
        module = module[len("src.") :]
    try:
        return importlib.import_module(module), None
    except Exception as err:  # noqa: BLE001 - report any import failure
        return None, err


def _has_submodule(module: str, name: str) -> bool:
    """Whether ``from module import name`` targets an existing submodule.

    ``importlib.util.find_spec`` is order-independent (unlike ``hasattr``,
    which only sees submodules already bound as package attributes). Returns
    False on any import failure so the caller falls back to a name check.
    """
    if module.startswith("src."):
        module = module[len("src.") :]
    if not module:
        return False
    try:
        spec = importlib.util.find_spec(f"{module}.{name}")
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
        return False
    return spec is not None


def _check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as err:
        return [f"{path}: PARSE ERROR: {err}"]

    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None or not _is_first_party(node.module):
                continue
            module_obj, exc = _resolve_module(node.module)
            if exc is not None or module_obj is None:
                issues.append(
                    f"{path}: import {node.module} -> {type(exc).__name__}: {exc}"
                )
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if _has_submodule(node.module, alias.name):
                    continue
                if not hasattr(module_obj, alias.name):
                    issues.append(f"{path}: name {alias.name!r} not in {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_first_party(alias.name):
                    continue
                module_obj, exc = _resolve_module(alias.name)
                if exc is not None or module_obj is None:
                    issues.append(
                        f"{path}: import {alias.name} -> {type(exc).__name__}: {exc}"
                    )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that all first-party imports/names in scripts/ and tests/ resolve."
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["scripts", "tests"],
        help="Directories to sweep (default: scripts tests).",
    )
    args = parser.parse_args()

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    issues: list[str] = []
    for rel in args.paths:
        base = ROOT_DIR / rel
        if not base.is_dir():
            print(f"[IMPORT-SWEEP] skip (not a directory): {rel}")
            continue
        for py_file in sorted(base.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            issues.extend(_check_file(py_file))

    if issues:
        print(f"[IMPORT-SWEEP] {len(issues)} broken first-party reference(s):")
        for issue in issues:
            print(f"  {issue}")
        print("[IMPORT-SWEEP] FAILED")
        return 1

    dirs = " ".join(args.paths)
    print(f"[IMPORT-SWEEP] OK: all first-party references resolve ({dirs})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
