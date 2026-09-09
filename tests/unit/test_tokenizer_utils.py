"""BM25 토크나이저 검증 — scripts/verification/verify_tokenizer.py 에서 마이그레이션.

원본은 print 기반이었으나, ``common.text_utils.bm25_tokenizer`` 가 주어진
한국어 입력에 대해 기대하는 토큰 집합을 만들고, 문서-쿼리 토큰 교집합으로
검색 재현율(recall) 시나리오가 동작하는지를 assert 로 검증합니다.
순수 로직으로 라이브 모델/네트워크 의존성 없음.
"""

from common.text_utils import bm25_tokenizer


def test_korean_compound_noun_preserved_with_bigrams() -> None:
    tokens = bm25_tokenizer("자연어처리는재미있다")

    # 전체 복합명사 보존 + bi-gram 생성 (검색 재현율 향상).
    assert "자연어처리는재미있다" in tokens
    assert "자연" in tokens
    assert "재미" in tokens
    assert "있다" in tokens


def test_korean_particle_stemming() -> None:
    tokens = bm25_tokenizer("학교에 간다")

    # 명사 + 조사("에") → 조사가 제거된 어간 "학교"가 추가된다.
    assert "학교에" in tokens
    assert "학교" in tokens


def test_mixed_korean_english_tokenization() -> None:
    tokens = bm25_tokenizer("RAG시스템 구축")

    # 영어는 소문자화, 한글은 명사/조사/bi-gram 처리.
    assert "rag" in tokens
    assert "시스템" in tokens
    assert "구축" in tokens


def test_empty_string_returns_empty_list() -> None:
    assert bm25_tokenizer("") == []


def test_original_sampling_cases_produce_tokens() -> None:
    test_cases = [
        "자연어처리는재미있다",  # 복합명사 + 형용사
        "데이터베이스에서검색한다",  # 명사 + 조사
        "RAG시스템 구축",  # 영어 + 한글 혼용
        "학교에 간다",  # 명사 + 조사
    ]

    for text in test_cases:
        tokens = bm25_tokenizer(text)
        assert tokens, f"no tokens produced for {text!r}"
        assert all(isinstance(t, str) and t for t in tokens)


def test_recall_scenario_doc_query_token_intersection() -> None:
    doc_text = "우리는 고성능 데이터베이스 시스템을 구축했습니다."
    doc_tokens = set(bm25_tokenizer(doc_text))

    queries = ["데이터", "베이스", "시스템", "구축"]
    for query in queries:
        query_tokens = set(bm25_tokenizer(query))
        assert doc_tokens.intersection(query_tokens), (
            f"no token match for query {query!r}: doc={doc_tokens} query={query_tokens}"
        )
