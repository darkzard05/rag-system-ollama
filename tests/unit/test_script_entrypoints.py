"""Path/bootstrap guard for standalone verification scripts.

Standalone scripts under ``scripts/`` are not collected by pytest or linted by
ruff/mypy, so path rot (e.g. a script moved into a subdirectory while keeping
``Path(__file__).parent.parent``) only surfaces at runtime. These tests import
the script module (which is deliberately cheap: heavy imports are lazy) and
assert that its repo-root asset anchors still resolve.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verification" / "verify_full_pipeline.py"


def load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_full_pipeline_script_anchors_resolve() -> None:
    assert SCRIPT_PATH.is_file()

    module = load_script(SCRIPT_PATH, "verify_full_pipeline_path_guard")

    assert module.ROOT_DIR == REPO_ROOT
    assert module.PDF_PATH.is_file()
    assert module.GOLDEN_PATH.is_file()
    assert (module.ROOT_DIR / "src" / "core" / "rag_core.py").is_file()


def test_full_pipeline_script_exposes_entrypoint() -> None:
    module = load_script(SCRIPT_PATH, "verify_full_pipeline_entrypoint_guard")

    assert callable(module.main)
    assert callable(module.parse_args)
