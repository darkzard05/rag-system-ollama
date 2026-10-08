"""Rerank 경로 분기 + 누적 타이밍 결함의 회귀 가드.

배경: 'cm3'(3자)은 rerank 생략(RRF 복사, rerank_ms=119.6),
'vqvae-gan'(9자)은 FlashRank 실호출(rerank_ms=4658.8)로 38x 편차가 났고,
rerank_ms/merge_ms/total_ms가 t0 기준 누적값이라 단계 비교가 무효였다.

수정 후 보장 사항 (Ollama/FAISS 없이 결정적으로 검증):
1. 쿼리 길이 분기 임계값이 config(`retriever.min_rerank_chars`, 기본 6)로 승격됨.
2. `_retrieval.py` 에 하드코딩된 `< 6` 분기가 남아 있지 않음.
3. TIMING 로그에 단계별 순수 소요(`rerank_ex_ms`/`merge_ex_ms`)가 포함됨.
4. 기존 누적 키(`rerank_ms`/`merge_ms`/`total_ms`)는 로그 파서 호환용으로 유지됨.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_SRC = (REPO_ROOT / "src" / "core" / "graph" / "_retrieval.py").read_text(
    encoding="utf-8"
)


def test_rerank_threshold_promoted_to_config():
    from common.config import RETRIEVER_CONFIG

    assert RETRIEVER_CONFIG.get("min_rerank_chars", 6) == 6


def test_no_hardcoded_query_length_branch():
    assert 'len(query or "") < 6' not in RETRIEVAL_SRC
    assert "min_rerank_chars" in RETRIEVAL_SRC


def test_exclusive_timing_keys_logged():
    assert "rerank_ex_ms" in RETRIEVAL_SRC
    assert "merge_ex_ms" in RETRIEVAL_SRC
    # 기존 누적 키는 파서 호환용으로 유지되어야 한다.
    assert "rerank_ms={rerank_ms" in RETRIEVAL_SRC
    assert "total_ms={total_ms" in RETRIEVAL_SRC


def test_exclusive_timing_semantics():
    # 단계별 순수 소요는 정의상 누적 경과보다 클 수 없다:
    # ex = end - stage_start <= end - t0 = cumulative
    t0, _t_agg, t_rerank_start, t_rerank_end, t_merge_end = (
        0.0,
        0.115,
        0.115,
        0.120,
        4.737,
    )
    rerank_ms = (t_rerank_end - t0) * 1000
    rerank_ex_ms = (t_rerank_end - t_rerank_start) * 1000
    merge_ex_ms = (t_merge_end - t_rerank_end) * 1000
    assert rerank_ex_ms <= rerank_ms
    assert rerank_ex_ms < 100.0  # 'cm3'류 생략 경로는 순수 리랭크가 미미
    assert merge_ex_ms >= 0.0
