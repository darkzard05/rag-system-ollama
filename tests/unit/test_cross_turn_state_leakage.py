"""B7: cross-turn state leakage 검증.

버그 배경: retrieve_and_rerank/grade 노드가 ``relevant_docs``를 턴 간 상태에 유지해,
문서 없음 턴에서 이전 턴의 문서가 프롬프트로 새는 문제. preprocess(턴 시작 노드)가
``relevant_docs=[]``로 리셋해 격리한다.
"""

from langchain_core.documents import Document

from core.graph.graph_builder import preprocess


class MockWriter:
    """preprocess의 StreamWriter 위치 인자를 대체하는 모의 객체."""

    def __call__(self, data: object, *, step: int | None = None) -> None:
        return None


def _doc(content: str) -> Document:
    return Document(page_content=content)


def _end_of_turn_state(docs: list[Document]) -> dict:
    """턴 1 종료 시점을 흉내 낸 상태 스냅샷 (검색 문서 + 응답 채워짐)."""
    return {
        "input": "이전 턴 질문",
        "intent": "rag",
        "route": "transform",
        "search_queries": ["previous rewritten query"],
        "relevant_docs": docs,
        "response": "이전 턴 응답",
        "retry_count": 2,
    }


async def test_preprocess_resets_documents_each_turn():
    """preprocess 출력 dict에 relevant_docs가 빈 리스트로 포함돼 턴 시작 리셋을 보낸다."""
    state = {"input": "새 질문: 어떻게 해야 하나요?"}
    config = {"configurable": {}}
    result = await preprocess(state, config, writer=MockWriter())

    assert result["relevant_docs"] == []
    # 다른 턴 리셋 키들은 계획된 값을 유지한다.
    assert result["search_queries"] == []
    assert result["retry_count"] == 0


async def test_no_docs_survive_second_turn():
    """턴 1 종료 상태에 턴 2 preprocess 출력을 병합하면 relevant_docs가 비워진다."""
    turn1_end = _end_of_turn_state([_doc("문서 A"), _doc("문서 B")])

    turn2_state = {"input": "새 질문: 문서 없음 시나리오"}
    turn2_config = {"configurable": {}}
    turn2_preprocess_out = await preprocess(
        turn2_state, turn2_config, writer=MockWriter()
    )

    merged = {**turn1_end, **turn2_preprocess_out}

    assert merged["relevant_docs"] == []
    assert merged["response"] == "이전 턴 응답"
