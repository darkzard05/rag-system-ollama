"""Sync machine-derived README sections with their source files."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
README_PATH = ROOT_DIR / "readme.md"

# Git pathspecs for the Last Updated date. readme.md is deliberately absent:
# including it would make the date chase its own writes and never converge.
# assets is absent too, so refreshing a screenshot does not invalidate the date.
GIT_DATE_PATHS = (
    "requirements/base.txt",
    "config.yml",
    "src/ui",
    "src/main.py",
)

LEGACY_EMBEDDING_MODEL = "nomic-embed-text"


class SyncError(RuntimeError):
    """A README marker or source is missing, duplicated, or unparseable."""


@dataclass(frozen=True)
class Context:
    repo: Path
    requirements: dict[str, str]
    config: dict[str, Any]
    coverage: int
    git_date: str


class Section(NamedTuple):
    name: str
    marker: str
    producer: Callable[[Context], str]
    sources: tuple[str, ...]


class SettingRow(NamedTuple):
    key: str
    source: str  # dotted config.yml path, or "=<literal>" when no config key
    purpose: str


TECH_PACKAGES: tuple[tuple[str, str], ...] = (
    ("Streamlit", "streamlit"),
    ("LangChain", "langchain"),
    ("LangGraph", "langgraph"),
    ("PyMuPDF4LLM", "pymupdf4llm"),
    ("Ollama", "ollama"),
    ("FastAPI", "fastapi"),
)

SETTING_ROWS: tuple[SettingRow, ...] = (
    SettingRow(
        "models.ollama_num_predict",
        "models.ollama_num_predict",
        "Max generated tokens per answer",
    ),
    SettingRow("models.num_ctx", "models.num_ctx", "Model context window (tokens)"),
    SettingRow(
        "models.thinking",
        "models.thinking",
        "Enable Ollama native reasoning extraction",
    ),
    SettingRow(
        "models.embedding_batch_size",
        "models.embedding_batch_size",
        "Embedding batch size (auto = GPU VRAM-based)",
    ),
    SettingRow(
        "models.allow_concurrent_inference",
        "models.allow_concurrent_inference",
        "T2 load-test gate: effective inference bound stays 1 until load gate passes",
    ),
    SettingRow(
        "rag.retriever.search_kwargs.k",
        "rag.retriever.search_kwargs.k",
        "Stage-1 candidate pool (matches `dynamic_top_k.max_candidates`)",
    ),
    SettingRow(
        "rag.retriever.min_rerank_chars",
        "rag.retriever.min_rerank_chars",
        "Queries shorter than this skip FlashRank and use RRF scores",
    ),
    SettingRow(
        "rag.retriever.rerank_top_k",
        "rag.prompts.grading.top_k",
        "Final rerank selection pool",
    ),
    SettingRow(
        "rag.prompts.grading.grade_top_n",
        "rag.prompts.grading.grade_top_n",
        "Docs passed to grading (widened 3→5: citations observed at rank 4–5)",
    ),
    SettingRow(
        "rag.parsing.hydration_mode",
        "rag.parsing.hydration_mode",
        "PDF coordinate extraction mode for highlighting",
    ),
    SettingRow(
        "rag.reranker.engine",
        "=auto",
        "Reranker engine: `auto` / `flashrank` / `semantic` (code default)",
    ),
    SettingRow(
        "rag.reranker.max_length",
        "rag.reranker.max_length",
        "FlashRank max sequence length (512 default; 384 cuts CPU rerank latency "
        "at equal golden-set context recall)",
    ),
    SettingRow(
        "rag.prompts.grading.min_score_to_skip",
        "rag.prompts.grading.min_score_to_skip",
        "Skip the LLM grade when the max rerank score reaches this "
        "(per-engine thresholds)",
    ),
    SettingRow(
        "rag.prompt_templates.citations.text_span",
        "=<=200 chars (advisory)",
        "Keep `text_span` under 200 chars; `maxLength: 200` is advisory only in "
        "`json_object` mode (not enforced by a validator)",
    ),
    SettingRow(
        "rag.prompts.generate.generation_validation",
        "rag.prompts.generation_validation.regen_on_short",
        "Generation validation flags",
    ),
    SettingRow(
        "global_cache.enable_vector_cache",
        "=true",
        "Enable vector store (indexing results) caching (code default)",
    ),
)

_REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(==|>=)\s*([^\s;#]+)")
_COVERAGE_RE = re.compile(r"^\s*COVERAGE_FAIL_UNDER\s*=\s*(\d+)", re.MULTILINE)
_EMBEDDING_RE = re.compile(
    r'DEFAULT_EMBEDDING_MODEL\s*:\s*str\s*=\s*_get_env\(\s*"[^"]*"\s*,\s*"([^"]+)"',
    re.DOTALL,
)


def _parse_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SyncError(f"cannot read {path}: {exc}") from exc
    for line in lines:
        match = _REQUIREMENT_RE.match(line)
        if match:
            name, operator, version = match.groups()
            pins[name.lower()] = version if operator == "==" else f"{operator}{version}"
    if not pins:
        raise SyncError(f"no == or >= pins found in {path}")
    return pins


def _parse_coverage(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SyncError(f"cannot read {path}: {exc}") from exc
    match = _COVERAGE_RE.search(text)
    if match is None:
        raise SyncError(f"COVERAGE_FAIL_UNDER not found in {path}")
    return int(match.group(1))


def _load_config(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SyncError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SyncError(f"{path} did not parse as a mapping")
    return data


def _git_date(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", *GIT_DATE_PATHS],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _build_context(repo: Path) -> Context:
    return Context(
        repo=repo,
        requirements=_parse_requirements(repo / "requirements" / "base.txt"),
        config=_load_config(repo / "config.yml"),
        coverage=_parse_coverage(repo / "scripts" / "maintenance" / "ci_parity.py"),
        git_date=_git_date(repo),
    )


def _count_tests(folder: Path) -> int:
    if not folder.is_dir():
        raise SyncError(f"missing test directory: {folder}")
    return sum(1 for path in folder.rglob("test_*.py") if path.is_file())


def _dotted(config: dict[str, Any], path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SyncError(f"config key missing: {path}")
        node = node[part]
    return node


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    raise SyncError(f"unsupported config value: {value!r}")


def _resolve_source(source: str, config: dict[str, Any]) -> str:
    if source.startswith("="):
        return source[1:]
    return _format_value(_dotted(config, source))


def _score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError) as exc:
        raise SyncError(f"score is not a number: {value!r}") from exc


def _tech_stack(ctx: Context) -> str:
    lines: list[str] = []
    for label, package in TECH_PACKAGES:
        pin = ctx.requirements.get(package)
        if pin is None:
            raise SyncError(f"missing requirement pin: {package}")
        lines.append(f"- **{label}**: {pin}")
    return "\n".join(lines)


def _test_counts(ctx: Context) -> str:
    unit = _count_tests(ctx.repo / "tests" / "unit")
    integration = _count_tests(ctx.repo / "tests" / "integration")
    return "\n".join(
        (
            f"- **{unit} unit test files** (plus {integration} integration test "
            "files) covering Graph Flow, RAG Orchestration, Document Processing, "
            "Advanced Citations, concurrency, and streaming.",
            "- **CI-enforced integrity:** `ruff check`, `ruff format --check`, "
            "`mypy src`, `bandit`, `pip-audit`, unit coverage "
            f"≥{ctx.coverage}%, and {integration} integration test files "
            "(auth/ownership/PDF/SSE).",
        )
    )


def _settings(ctx: Context) -> str:
    rows: list[str] = [
        "| Key | Default | Purpose |",
        "|-----|---------|---------|",
    ]
    for row in SETTING_ROWS:
        purpose = row.purpose
        if row.key == "rag.prompts.grading.min_score_to_skip":
            flash = _score(_dotted(ctx.config, row.source))
            semantic = _score(
                _dotted(ctx.config, "rag.prompts.grading.min_score_to_skip_semantic")
            )
            value = f"{flash} (FlashRank) / {semantic} (semantic)"
        elif row.key == "rag.prompts.generate.generation_validation":
            root = row.source.rsplit(".", 1)[0]
            value = f"regen_on_short={_resolve_source(row.source, ctx.config)}"
            over = _resolve_source(f"{root}.over_budget_chars", ctx.config)
            floor = _resolve_source(f"{root}.regen_floor_chars", ctx.config)
            purpose = (
                "S1 telemetry flags (over_budget_chars: "
                f"{over}, regen_floor_chars: {floor})"
            )
        else:
            value = _resolve_source(row.source, ctx.config)
        rows.append(f"| `{row.key}` | {value} | {purpose} |")
    return "\n".join(rows)


def _embedding_default(repo: Path) -> str:
    path = repo / "src" / "common" / "config.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SyncError(f"cannot read {path}: {exc}") from exc
    match = _EMBEDDING_RE.search(text)
    if match is None:
        raise SyncError(f"DEFAULT_EMBEDDING_MODEL literal not found in {path}")
    return match.group(1)


def _badges(ctx: Context) -> str:
    model = _dotted(ctx.config, "models.default_ollama")
    if not isinstance(model, str) or not model:
        raise SyncError("models.default_ollama must be a non-empty string")
    badge_model = model.replace("-", "--")
    library = model.split(":", 1)[0]
    embedding = _embedding_default(ctx.repo)
    return "\n".join(
        (
            f"[![Model: {model}](https://img.shields.io/badge/Model-"
            f"{badge_model}-blueviolet.svg)](https://ollama.com/library/{library})",
            "```powershell",
            "# Pull the recommended models",
            f"ollama pull {model}",
            f"ollama pull {embedding}  # default embedding model "
            "(src/common/config.py: DEFAULT_EMBEDDING_MODEL)",
            "# Optional (legacy embedding model):",
            f"ollama pull {LEGACY_EMBEDDING_MODEL}",
            "```",
        )
    )


def _last_updated(ctx: Context) -> str:
    return f"**Last Updated:** {ctx.git_date}"


SECTIONS: tuple[Section, ...] = (
    Section(
        "Tech Stack",
        "TECH_STACK",
        _tech_stack,
        ("requirements/base.txt",),
    ),
    Section(
        "Test Counts",
        "TEST_COUNTS",
        _test_counts,
        (
            "tests/unit",
            "tests/integration",
            "scripts/maintenance/ci_parity.py",
        ),
    ),
    Section(
        "Key Settings",
        "SETTINGS",
        _settings,
        ("config.yml", "src/common/config.py"),
    ),
    Section(
        "Model Badges",
        "BADGES",
        _badges,
        ("config.yml", "src/common/config.py"),
    ),
    Section("Last Updated", "LAST_UPDATED", _last_updated, GIT_DATE_PATHS),
)


def _marker_pair(section: Section) -> tuple[str, str]:
    return (
        f"<!-- {section.marker}_START -->",
        f"<!-- {section.marker}_END -->",
    )


def _validate_markers(content: str) -> None:
    for section in SECTIONS:
        start, end = _marker_pair(section)
        for marker in (start, end):
            count = content.count(marker)
            if count != 1:
                raise SyncError(f"{section.name}: expected one {marker}, found {count}")


def _block_pattern(section: Section) -> re.Pattern[str]:
    start, end = _marker_pair(section)
    return re.compile(f"{re.escape(start)}(.*?){re.escape(end)}", re.DOTALL)


def _block_text(content: str, section: Section) -> str:
    match = _block_pattern(section).search(content)
    if match is None:
        raise SyncError(f"{section.name}: markers are out of order")
    return match.group(1).strip("\n")


def _replace_block(content: str, section: Section, rendered: str) -> str:
    start, end = _marker_pair(section)
    pattern = re.compile(
        f"({re.escape(start)}).*?({re.escape(end)})",
        re.DOTALL,
    )

    def replacement(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n{rendered}\n{match.group(2)}"

    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise SyncError(f"{section.name}: markers are out of order")
    return updated


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_path, path)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def sync_readme(root: Path, check: bool = False) -> list[str]:
    """Return drifted section labels in check mode; write converged content otherwise."""
    readme = root / "readme.md"
    if not readme.is_file():
        raise SyncError(f"README not found: {readme}")
    content = readme.read_text(encoding="utf-8")
    _validate_markers(content)

    for section in SECTIONS:
        missing = [source for source in section.sources if not (root / source).exists()]
        if missing:
            raise SyncError(f"{section.name}: missing source(s): {', '.join(missing)}")

    ctx = _build_context(root)
    expected = content
    drifted: list[str] = []
    for section in SECTIONS:
        rendered = section.producer(ctx)
        expected = _replace_block(expected, section, rendered)
        if check and _block_text(content, section) != rendered:
            drifted.append(f"{section.name} ({section.marker})")

    if check:
        return drifted
    if expected != content:
        _atomic_write(readme, expected)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true", help="Write synced content (default)."
    )
    mode.add_argument("--check", action="store_true", help="Report drift, never write.")
    parser.add_argument(
        "--root", type=Path, default=ROOT_DIR, help="Repo root (defaults to this repo)."
    )
    args = parser.parse_args(argv)

    try:
        drifted = sync_readme(args.root, check=args.check)
    except SyncError as exc:
        print(f"update_readme: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if drifted:
            for section in drifted:
                print(f"drift: {section}")
            return 1
        print("readme.md is in sync")
        return 0
    print("readme.md synced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
