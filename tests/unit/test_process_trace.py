"""T2 _build_process 요약 사전 생성 검증 (answer-process-expander).

스트리밍 중 누적한 process_steps/documents/metrics를 완료 시점에 요약하는
순수 헬퍼 `_build_process`의 단계 중복 제거·캡핑, 문서 메타데이터 이중 처리
(langchain Document 객체 / 일반 dict), rerank 상위 3개, metrics 키 추출을
검증합니다.
"""

from ui.components.streaming import _build_process


class _Doc:
    """langchain Document(m .metadata 속성) 경량 스텁 클래스."""

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


def _steps_msg(process_steps: list[str]) -> dict:
    return {"process_steps": process_steps}


def test_steps_order_preserved_and_consecutive_dedup():
    p = _build_process(_steps_msg(["A", "A", "B"]))
    assert p["steps"] == ["A", "B"]


def test_steps_non_consecutive_repeat_kept():
    p = _build_process(_steps_msg(["A", "B", "A"]))
    assert p["steps"] == ["A", "B", "A"]


def test_steps_capped_to_last_10():
    steps_input = [f"s{i}" for i in range(14)]
    p = _build_process(_steps_msg(steps_input))
    assert p["steps"] == [f"s{i}" for i in range(4, 14)]


def test_steps_empty_without_process_steps():
    p = _build_process({})
    assert p["steps"] == []


def test_retrieved_count_matches_documents():
    docs = [{"metadata": {}}, {"metadata": {}}, {"metadata": {}}]
    p = _build_process(_steps_msg([]) | {"documents": docs})
    assert p["retrieved_count"] == 3


def test_sections_dedup_order_preserving_max_5_with_doc_objects():
    docs = [
        _Doc({"current_section": "Intro"}),
        _Doc({"current_section": "Intro"}),
        _Doc({"current_section": "Methods"}),
        _Doc({}),
        _Doc({"current_section": "Results"}),
        _Doc({"current_section": "Discussion"}),
        _Doc({"current_section": "Limits"}),
        _Doc({"current_section": "Conclusion"}),
        _Doc({"current_section": "Appendix"}),
    ]
    p = _build_process({"documents": docs})
    assert p["sections"] == ["Intro", "Methods", "Results", "Discussion", "Limits"]


def test_sections_works_with_dicts_and_max_5():
    docs = [{"metadata": {"current_section": f"Sec{i}"}} for i in range(8)]
    p = _build_process({"documents": docs})
    assert p["sections"] == [f"Sec{i}" for i in range(5)]


def test_sections_skips_falsy_section():
    docs = [
        _Doc({"current_section": ""}),
        _Doc({"current_section": None}),
        _Doc({"current_section": "Real"}),
    ]
    p = _build_process({"documents": docs})
    assert p["sections"] == ["Real"]


def test_top_scores_desc_top3_rounded():
    docs = [
        {"metadata": {"current_section": "A", "rerank_score": 1.23456}},
        {"metadata": {"current_section": "B", "rerank_score": 5.99999}},
        {"metadata": {"current_section": "C", "rerank_score": 3.14159}},
        {"metadata": {"current_section": "D", "rerank_score": 0.00001}},
    ]
    p = _build_process({"documents": docs})
    assert p["top_scores"] == [
        {"section": "B", "score": 6.0},
        {"section": "C", "score": 3.142},
        {"section": "A", "score": 1.235},
    ]


def test_top_scores_skips_none_missing_and_non_numeric():
    docs = [
        _Doc({"current_section": "A", "rerank_score": None}),
        _Doc({"current_section": "B", "rerank_score": "n/a"}),
        _Doc({"current_section": "C"}),
        _Doc({"current_section": "D", "rerank_score": 2.0}),
    ]
    p = _build_process({"documents": docs})
    assert p["top_scores"] == [{"section": "D", "score": 2.0}]


def test_top_scores_empty_when_all_lack_scores():
    docs = [
        _Doc({"current_section": "A"}),
        {"metadata": {"current_section": "B", "rerank_score": None}},
    ]
    p = _build_process({"documents": docs})
    assert p["top_scores"] == []


def test_perf_only_allowed_keys_and_ttft_excluded():
    metrics = {
        "total_time": 1.5,
        "tps": 30.0,
        "input_token_count": 100,
        "token_count": 200,
        "relevant_docs_count": 3,
        "ttft": 0.5,
        "extra_key": 99,
    }
    p = _build_process({"metrics": metrics})
    assert p["perf"] == {
        "total_time": 1.5,
        "tps": 30.0,
        "input_token_count": 100,
        "token_count": 200,
        "relevant_docs_count": 3,
    }
    assert "ttft" not in p["perf"]


def test_perf_missing_keys_absent():
    metrics = {"total_time": 1.5, "ttft": 0.5}
    p = _build_process({"metrics": metrics})
    assert p["perf"] == {"total_time": 1.5}
    assert "ttft" not in p["perf"]


def test_empty_inputs_all_keys_with_defaults():
    p = _build_process({})
    assert p == {
        "steps": [],
        "retrieved_count": 0,
        "sections": [],
        "top_scores": [],
        "perf": {},
    }


def test_build_process_does_not_mutate_msg():
    msg = {
        "process_steps": ["A", "A"],
        "documents": [{"metadata": {"current_section": "X"}}],
        "metrics": {"total_time": 1.0},
    }
    snapshot = {"process_steps": list(msg["process_steps"]), **msg}
    _build_process(msg)
    assert msg["process_steps"] == ["A", "A"]
    assert msg["documents"] == snapshot["documents"]
    assert msg["metrics"] == {"total_time": 1.0}
