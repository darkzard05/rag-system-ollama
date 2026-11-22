"""Drift guards for the CI <-> pre-push parity contract.

The whole "pre-push green => CI green" claim rests on both entry points running
the SAME commands. These tests fail if someone re-inlines a gating command into
ci.yml, adds a second pre-push hook, lowers the coverage gate, or deletes a
gating test file from the single source of truth.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_PY = ROOT / "scripts" / "maintenance" / "ci_parity.py"


def _load_ci_parity() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ci_parity", CI_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_parity"] = module
    spec.loader.exec_module(module)
    return module


ci_parity = _load_ci_parity()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_blocks(workflow: dict) -> list[str]:
    blocks: list[str] = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run")
            if isinstance(run, str):
                blocks.append(run)
    return blocks


def test_coverage_gate_is_70() -> None:
    assert ci_parity.COVERAGE_FAIL_UNDER == 70


def test_gating_files_exist() -> None:
    missing = [
        p for p in ci_parity.GATING_INTEGRATION_FILES if not (ROOT / p).is_file()
    ]
    assert not missing, f"gating files missing: {missing}"
    missing_real = [p for p in ci_parity.REAL_MODEL_FILES if not (ROOT / p).is_file()]
    assert not missing_real, f"real-model files missing: {missing_real}"


def test_gating_list_has_no_duplicates() -> None:
    files = ci_parity.GATING_INTEGRATION_FILES
    assert len(files) == len(set(files))


def test_ci_yml_delegates_every_gating_group() -> None:
    workflow = _load_yaml(ROOT / ".github" / "workflows" / "ci.yml")
    runs = "\n".join(_run_blocks(workflow))
    for group in ("static", "security", "tests"):
        assert re.search(rf"ci_parity\.py\s+--only\s+{group}\b", runs), (
            f"ci.yml does not delegate the '{group}' group to ci_parity.py"
        )


_RAW_GATE = re.compile(
    r"^\s*(?:python\s+-m\s+)?(?:pytest|ruff|mypy|bandit|pip-audit)\b"
)


def test_ci_yml_has_no_inline_gating_commands() -> None:
    workflow = _load_yaml(ROOT / ".github" / "workflows" / "ci.yml")
    offenders: list[str] = []
    for block in _run_blocks(workflow):
        for line in block.splitlines():
            if _RAW_GATE.match(line):
                offenders.append(line.strip())
    assert not offenders, (
        "ci.yml inlines gating commands; move them into ci_parity.py so the "
        f"pre-push hook cannot drift: {offenders}"
    )


def test_pre_push_has_only_the_parity_hooks() -> None:
    config = _load_yaml(ROOT / ".pre-commit-config.yaml")
    ids: list[str] = []
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            stages = hook.get("stages") or []
            if "pre-push" in stages:
                ids.append(hook["id"])
    assert sorted(ids) == ["ci-parity", "verify-requirements"], (
        "pre-push must run exactly the ci-parity gate (+ fast requirements "
        f"check); found: {ids}"
    )


def test_pre_push_ci_parity_uses_the_pre_push_stage() -> None:
    config = _load_yaml(ROOT / ".pre-commit-config.yaml")
    entries = [
        hook["entry"]
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "ci-parity"
    ]
    assert len(entries) == 1
    assert "--stage pre-push" in entries[0]


def test_mypy_uses_sqlite_cache() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"^sqlite_cache\s*=\s*true", text, re.MULTILINE)


def test_pytest_has_hard_timeout() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"^timeout\s*=\s*\d+", text, re.MULTILINE)


def test_changed_file_scoping_matches_ci_trigger_paths() -> None:
    workflow = _load_yaml(ROOT / ".github" / "workflows" / "ci.yml")
    # PyYAML resolves the bare `on:` key to boolean True.
    triggers = workflow.get("on") or workflow.get(True)
    assert isinstance(triggers, dict), "ci.yml has no on: trigger block"
    ci_paths: set[str] = set()
    for event in triggers.values():
        if isinstance(event, dict):
            ci_paths.update(event.get("paths", []))

    prefixes = {p.rstrip("*").rstrip("/") + "/" for p in ci_paths if p.endswith("/**")}
    missing = prefixes - set(ci_parity.TRIGGER_PREFIXES)
    assert not missing, (
        "ci_parity would skip a push that CI runs: missing trigger prefixes "
        f"{sorted(missing)}"
    )


@pytest.mark.parametrize(
    "step_group", ["static", "security", "tests", "extended", "real"]
)
def test_every_group_resolves_to_steps(step_group: str) -> None:
    steps = ci_parity._build_steps()
    assert [s for s in steps if s.group == step_group]
