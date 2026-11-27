"""Observable behavior of the README sync CLI and its failure modes."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "update_readme.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_update_readme_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


update_readme = _load_module()

README = """# Demo

<!-- TECH_STACK_START -->
managed
<!-- TECH_STACK_END -->

<!-- TEST_COUNTS_START -->
managed
<!-- TEST_COUNTS_END -->

<!-- SETTINGS_START -->
managed
<!-- SETTINGS_END -->

<!-- BADGES_START -->
managed
<!-- BADGES_END -->

<!-- LAST_UPDATED_START -->
managed
<!-- LAST_UPDATED_END -->
"""

REQUIREMENTS = """streamlit==1.66.1
langchain>=1.3.0
langgraph>=1.2.0
pymupdf4llm>=0.3.4
ollama==0.6.1
fastapi==0.133.1
"""

CONFIG = """models:
  default_ollama: "qwen3:4b-instruct-2507-q4_K_M"
  ollama_num_predict: 2048
  num_ctx: 10240
  thinking: false
  embedding_batch_size: "auto"
  allow_concurrent_inference: false
rag:
  retriever:
    min_rerank_chars: 6
    search_kwargs:
      k: 18
    dynamic_top_k:
      max_candidates: 18
  parsing:
    hydration_mode: "precision_clip"
  reranker:
    max_length: 384
  prompts:
    generation_validation:
      over_budget_chars: 1400
      regen_floor_chars: 700
      regen_on_short: true
    grading:
      top_k: 10
      grade_top_n: 5
      min_score_to_skip: 0.70
      min_score_to_skip_semantic: 0.45
"""

CONFIG_PY = """DEFAULT_EMBEDDING_MODEL: str = _get_env(
    "DEFAULT_EMBEDDING_MODEL",
    "nomic-embed-text-v2-moe",
)
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "readme.md").write_text(README, encoding="utf-8")
    (root / "requirements").mkdir(parents=True)
    (root / "requirements" / "base.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (root / "config.yml").write_text(CONFIG, encoding="utf-8")
    parity = root / "scripts" / "maintenance" / "ci_parity.py"
    parity.parent.mkdir(parents=True)
    parity.write_text("COVERAGE_FAIL_UNDER = 70\n", encoding="utf-8")
    config_py = root / "src" / "common" / "config.py"
    config_py.parent.mkdir(parents=True)
    config_py.write_text(CONFIG_PY, encoding="utf-8")
    (root / "src" / "main.py").write_text("", encoding="utf-8")
    (root / "src" / "ui").mkdir()
    (root / "assets").mkdir()
    unit = root / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_one.py").write_text("def test_one(): pass\n", encoding="utf-8")
    integration = root / "tests" / "integration"
    integration.mkdir(parents=True)
    (integration / "test_two.py").write_text("def test_two(): pass\n", encoding="utf-8")
    return root


def _run(root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), mode, "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_write_converges_on_second_run(repo: Path) -> None:
    first = _run(repo, "--write")
    assert first.returncode == 0, first.stderr
    after_first = (repo / "readme.md").read_bytes()
    assert b"managed" not in after_first

    second = _run(repo, "--write")
    assert second.returncode == 0, second.stderr
    assert (repo / "readme.md").read_bytes() == after_first


def test_check_is_clean_after_write(repo: Path) -> None:
    assert _run(repo, "--write").returncode == 0
    before = (repo / "readme.md").read_bytes()

    result = _run(repo, "--check")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (repo / "readme.md").read_bytes() == before


def test_check_names_drifted_section_without_writing(repo: Path) -> None:
    assert _run(repo, "--write").returncode == 0
    before = (repo / "readme.md").read_bytes()
    requirements = repo / "requirements" / "base.txt"
    requirements.write_text(
        requirements.read_text(encoding="utf-8").replace(
            "streamlit==1.66.1", "streamlit==9.9.9"
        ),
        encoding="utf-8",
    )

    result = _run(repo, "--check")

    assert result.returncode == 1
    assert "TECH_STACK" in result.stdout
    assert (repo / "readme.md").read_bytes() == before


def test_missing_marker_raises_without_writing(repo: Path) -> None:
    readme = repo / "readme.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "<!-- TEST_COUNTS_END -->", "<!-- TEST_COUNTS_GONE -->"
        ),
        encoding="utf-8",
    )
    before = readme.read_bytes()

    with pytest.raises(update_readme.SyncError):
        update_readme.sync_readme(repo)

    assert readme.read_bytes() == before


def test_generated_values_come_from_sources(repo: Path) -> None:
    assert _run(repo, "--write").returncode == 0
    text = (repo / "readme.md").read_text(encoding="utf-8")

    assert "- **Streamlit**: 1.66.1" in text
    assert "- **PyMuPDF4LLM**: >=0.3.4" in text
    assert "**1 unit test files** (plus 1 integration test files)" in text
    assert "unit coverage ≥70%" in text
    assert "| `models.thinking` | false |" in text
    assert "| `rag.retriever.rerank_top_k` | 10 |" in text
    assert (
        "| `rag.prompts.grading.min_score_to_skip` | "
        "0.70 (FlashRank) / 0.45 (semantic) | Skip the LLM grade when"
    ) in text
    assert (
        "| `rag.prompts.generate.generation_validation` | regen_on_short=true |" in text
    )
    assert "over_budget_chars: 1400, regen_floor_chars: 700" in text
    assert "ollama pull nomic-embed-text-v2-moe" in text
