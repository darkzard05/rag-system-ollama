"""preprocess 의도 분류 동적 가중치 판정 검증.

버그 배경: 정의형 질문("cm3가 뭔가요?")이 용어("cm3")만 매칭되어 keyword-heavy
(BM25 0.8)로 분류됐다. semantic_keywords에 한국어 질문형("뭔가요" 등)을 추가하면
keyword+semantic 동시 매칭 시 폴백되어 기본 가중치(ENSEMBLE_WEIGHTS)를 사용한다.
"""

from core.graph_builder import preprocess


class MockWriter:
    """preprocess의 StreamWriter 키워드 인자를 대체하는 모의 객체."""

    def __call__(self, data: object, *, step: int | None = None) -> None:
        return None


async def _preprocess_weights(query: str) -> dict[str, float]:
    state = {"input": query}
    config = {"configurable": {}}
    result = await preprocess(state, config, writer=MockWriter())
    return result["search_weights"]


async def test_definitional_question_both_match_falls_back_to_default():
    """정의형 질문("cm3가 뭔가요?"): keyword+semantic 동시 매칭 → 기본 가중치 폴백."""
    weights = await _preprocess_weights("cm3가 뭔가요?")
    assert weights == {"bm25": 0.4, "faiss": 0.6}


async def test_semantic_only_question_uses_semantic_weight():
    """의미 중심 질문: semantic만 매칭 → {bm25: 0.2, faiss: 0.8}."""
    weights = await _preprocess_weights("이 문서의 구조를 설명해줘")
    assert weights == {"bm25": 0.2, "faiss": 0.8}


async def test_keyword_only_query_uses_keyword_weight():
    """용어 중심 질문: keyword만 매칭([a-zA-Z]+\\d+) → {bm25: 0.8, faiss: 0.2}."""
    weights = await _preprocess_weights("CM3 3.2절")
    assert weights == {"bm25": 0.8, "faiss": 0.2}


async def test_no_match_falls_back_to_default():
    """어느 쪽도 매칭되지 않으면 기본 가중치(ENSEMBLE_WEIGHTS)를 사용한다."""
    weights = await _preprocess_weights("일반적인 문장입니다")
    assert weights == {"bm25": 0.4, "faiss": 0.6}
