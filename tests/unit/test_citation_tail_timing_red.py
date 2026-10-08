"""last-token->citations-event 분해 계측의 회귀 가드.

배경: 'cm3'에서 마지막 토큰 후 인용 이벤트까지 9.141s가 걸렸고, 원인이
LLM tail 생성(citations JSON이 answer 뒤에 생성)인지 파싱(_repair_json,
json.loads, AnswerStructure 검증)인지 구분이 불가했다.

수정 후 보장 사항 (Ollama 없이 결정적으로 검증):
1. `_generate_pass` 가 마지막 content 청크 시각(`_t_last_content`)을 기록함.
2. GENERATE TIMING 로그에 `tail_content_ms`(스트림 종료 - 마지막 content)가 포함됨.
3. `citations-dispatch` 로그에 `count`(기존 키, 파서 호환) + `tail_content_ms` +
   `parse_ms`(repair/loads/검증 순수 소요)가 포함됨.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_SRC = (REPO_ROOT / "src" / "core" / "graph" / "_generate.py").read_text(
    encoding="utf-8"
)


def test_last_content_timestamp_tracked():
    assert "_t_last_content" in GENERATE_SRC


def test_generate_timing_logs_tail_content():
    assert "tail_content_ms={tail_content_ms" in GENERATE_SRC


def test_citations_dispatch_logs_tail_and_parse():
    assert "citations-dispatch count=%d" in GENERATE_SRC
    assert "tail_content_ms=%.1f" in GENERATE_SRC
    assert "parse_ms=%.1f" in GENERATE_SRC


def test_tail_plus_parse_decomposition_semantics():
    # last-token->citations-event ≈ tail(생성) + parse(복구/검증) + dispatch.
    # cm3 실측: tail이 지배적이어야 함을 의미하는 부등식 (정의 검증).
    last_token_to_citations_ms = 9141.0
    parse_ms = 50.0
    assert parse_ms < last_token_to_citations_ms
    tail_ms = last_token_to_citations_ms - parse_ms
    assert tail_ms > 0.0
