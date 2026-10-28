"""Cold-start import isolation regressions (P0/P1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_core_session_import_does_not_load_rag_stack() -> None:
    """Importing core.session must not pull LangGraph, langchain_ollama, or
    transformers onto the first-paint path; graph_builder stays lazy."""
    code = """
import sys
sys.path.insert(0, "src")
import core.session
for prefix in ("langgraph", "langchain_ollama", "transformers"):
    loaded = [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]
    assert not loaded, loaded
from core import graph_builder
assert hasattr(graph_builder, "build_graph")
"""
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


def test_main_module_level_does_not_load_transformers() -> None:
    source = (_PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
    module_level = source.split("def main() -> None:", 1)[0]
    assert "silence_transformers_auto_docstring" not in module_level


def test_huggingface_loader_silences_before_transformers_import() -> None:
    source = (_PROJECT_ROOT / "src" / "core" / "model_loader.py").read_text(
        encoding="utf-8"
    )
    silencer = "silence_transformers_auto_docstring()"
    hf_import = "from langchain_huggingface import HuggingFaceEmbeddings"
    assert silencer in source
    assert source.index(silencer) < source.index(hf_import)
