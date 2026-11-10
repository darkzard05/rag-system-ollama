"""Pure-logic tests for scripts/evaluation/eval_qa_trace.py (no LLM calls)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "eval_qa_trace.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_qa_trace", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_qa_trace"] = module
    spec.loader.exec_module(module)
    return module


eval_qa_trace = _load_module()


def test_load_trace_skips_blank_and_malformed(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"query": "q1"}),
                "",
                "{not json",
                json.dumps({"query": "q2"}),
                "[1, 2]",
            ]
        ),
        encoding="utf-8",
    )
    records, malformed = eval_qa_trace.load_trace(path)
    assert [r["query"] for r in records] == ["q1", "q2"]
    assert malformed == 2


def test_load_testset_normalizes_questions(tmp_path: Path) -> None:
    path = tmp_path / "testset.csv"
    path.write_text(
        "question,ground_truth,reference_context\n"
        '"  What   is A? ","answer A","ctx A"\n',
        encoding="utf-8",
    )
    mapping = eval_qa_trace.load_testset(path)
    assert mapping["what is a?"]["ground_truth"] == "answer A"
    assert mapping["what is a?"]["reference_context"] == "ctx A"


def test_load_testset_handles_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_text(
        'question,ground_truth,reference_context\n"BOM 질문","정답","근거"\n',
        encoding="utf-8-sig",
    )
    mapping = eval_qa_trace.load_testset(path)
    assert mapping["bom 질문"]["ground_truth"] == "정답"


def test_load_testset_missing_file_returns_empty(tmp_path: Path) -> None:
    assert eval_qa_trace.load_testset(tmp_path / "missing.csv") == {}


def test_attach_ground_truth_matches_normalized_query() -> None:
    records = [
        {"query": "What  is A?"},
        {"query": "unknown"},
    ]
    testset = {"what is a?": {"ground_truth": "gt", "reference_context": "ctx"}}
    joined = eval_qa_trace.attach_ground_truth(records, testset)
    assert joined == 1
    assert records[0]["ground_truth"] == "gt"
    assert records[0]["reference_context"] == "ctx"
    assert records[1]["ground_truth"] == ""


def test_parse_score() -> None:
    assert eval_qa_trace.parse_score("5") == 5
    assert eval_qa_trace.parse_score("점수: 3점") == 3
    assert eval_qa_trace.parse_score("no number here") is None


def test_filter_records_applies_filters_and_keeps_recent() -> None:
    records = [
        {"query": "q1", "is_cached": False, "intent": "rag"},
        {"query": "q2", "is_cached": True, "intent": "rag"},
        {"query": "q3", "is_cached": False, "intent": "general"},
        {"query": "q4", "is_cached": False, "intent": "rag"},
    ]
    assert [r["query"] for r in eval_qa_trace.filter_records(records)] == [
        "q1",
        "q2",
        "q3",
        "q4",
    ]
    assert [
        r["query"] for r in eval_qa_trace.filter_records(records, exclude_cached=True)
    ] == ["q1", "q3", "q4"]
    assert [
        r["query"] for r in eval_qa_trace.filter_records(records, intent="rag")
    ] == ["q1", "q2", "q4"]
    assert [r["query"] for r in eval_qa_trace.filter_records(records, limit=2)] == [
        "q3",
        "q4",
    ]


def test_aggregate_computes_averages_and_counts() -> None:
    records = [
        {
            "judge": 5,
            "is_cached": False,
            "source": "aquery",
            "intent": "rag",
            "answer": "aaaa",
            "parse_failed": False,
            "echo_detected": False,
            "over_budget": False,
            "judge_excluded": False,
        },
        {
            "judge": 3,
            "is_cached": True,
            "source": "astream",
            "intent": "rag",
            "answer": "aa",
            "parse_failed": True,
            "echo_detected": False,
            "over_budget": True,
            "judge_excluded": False,
        },
        {
            "judge": None,
            "is_cached": False,
            "source": "astream",
            "intent": "",
            "answer": "a",
            "parse_failed": False,
            "echo_detected": True,
            "over_budget": False,
            "judge_excluded": True,
        },
    ]
    agg = eval_qa_trace.aggregate(records)
    assert agg["record_count"] == 3
    assert agg["judged_count"] == 2
    assert agg["judge_avg"] == 4.0
    assert agg["score_distribution"] == {
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 0,
        "5": 1,
    }
    assert agg["excluded_count"] == 1
    assert agg["parse_failed_count"] == 1
    assert agg["echo_detected_count"] == 1
    assert agg["over_budget_count"] == 1
    assert agg["cached_count"] == 1
    assert agg["avg_answer_chars"] == round(7 / 3, 4)
    assert agg["cached_judge_avg"] == 3.0
    assert agg["uncached_judge_avg"] == 5.0
    assert agg["by_source"] == {"astream": 3.0, "aquery": 5.0}
    assert agg["by_intent"] == {"rag": 4.0}


def test_write_report_emits_json_and_markdown(tmp_path: Path) -> None:
    records = [
        {
            "query": "q|with|pipes",
            "answer": "a",
            "judge": 5,
            "judge_error": "",
            "source": "aquery",
            "intent": "rag",
            "is_cached": False,
        }
    ]
    agg = eval_qa_trace.aggregate(records)
    meta = {
        "trace_path": "trace.jsonl",
        "generated_at": "2026-01-01T00:00:00",
        "judge": "off",
        "judge_model": "m",
        "joined": 0,
        "malformed": 0,
    }
    json_path, md_path = eval_qa_trace.write_report(records, agg, meta, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["judge_avg"] == 5.0
    markdown = md_path.read_text(encoding="utf-8")
    assert "# QA Trace Evaluation" in markdown
    assert "q\\|with\\|pipes" in markdown


def test_judge_records_scores_and_excludes(monkeypatch) -> None:
    from langchain_core import prompts

    import core.model_loader as model_loader

    class _FakeChain:
        async def ainvoke(self, payload):
            return "4"

        def __or__(self, other):
            return self

    class _FakePrompt:
        def __or__(self, other):
            return _FakeChain()

    monkeypatch.setattr(
        prompts.ChatPromptTemplate,
        "from_template",
        lambda *args, **kwargs: _FakePrompt(),
    )
    monkeypatch.setattr(
        model_loader.ModelManager, "get_llm", AsyncMock(return_value=object())
    )
    records = [
        {
            "query": "q1",
            "ground_truth": "gt",
            "answer": "a",
            "reference_context": "ctx",
        },
        {"query": "q2", "ground_truth": "", "answer": "b"},
    ]
    asyncio.run(eval_qa_trace.judge_records(records, model_name="m"))
    assert records[0]["judge"] == 4
    assert records[0]["judge_excluded"] is False
    assert records[1]["judge"] is None
    assert records[1]["judge_excluded"] is True
    assert "join miss" in records[1]["judge_error"]


def test_main_no_judge_writes_report(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"query": "q", "answer": "a"}), encoding="utf-8")
    code = eval_qa_trace.main(
        [
            "--trace",
            str(trace),
            "--no-judge",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    outputs = list((tmp_path / "out").glob("eval_qa_trace_*.json"))
    assert len(outputs) == 1


def test_render_markdown_has_per_record_table() -> None:
    markdown = eval_qa_trace.render_markdown(
        {
            "trace_path": "t",
            "generated_at": "now",
            "judge": "on",
            "judge_model": "m",
            "joined": 1,
            "malformed": 0,
        },
        eval_qa_trace.aggregate([]),
        [],
    )
    assert "## Per record" in markdown
    assert "| judged_count | 0 |" in markdown
