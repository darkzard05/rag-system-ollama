from langchain_core.documents import Document
from src.core.graph_builder import _merge_consecutive_chunks, format_context


def test_metadata_to_context_string_conversion():
    """
    메타데이터(페이지 번호)가 LLM용 컨텍스트 문자열로 잘 변환되는지 테스트합니다.
    """
    # 1. 테스트용 문서 데이터 준비 (페이지 1에 2개 청크, 페이지 2에 1개 청크)
    docs = [
        Document(
            page_content="첫 번째 페이지의 첫 내용입니다.",
            metadata={"source": "test.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="첫 번째 페이지의 이어진 내용입니다.",
            metadata={"source": "test.pdf", "page": 1, "chunk_index": 1},
        ),
        Document(
            page_content="두 번째 페이지의 독립된 내용입니다.",
            metadata={"source": "test.pdf", "page": 2, "chunk_index": 0},
        ),
    ]

    print("\n[Test] 원본 문서 메타데이터 확인...")
    for i, d in enumerate(docs):
        print(
            f"  Doc {i}: Page {d.metadata['page']}, Index {d.metadata['chunk_index']}"
        )

    # 2. 청크 병합 테스트
    merged_docs = _merge_consecutive_chunks(docs)
    print(f"\n[Test] 병합 후 문서 개수: {len(merged_docs)} (기대값: 2)")

    assert len(merged_docs) == 2
    assert "이어진 내용" in merged_docs[0].page_content
    assert merged_docs[0].metadata["page"] == 1
    assert merged_docs[1].metadata["page"] == 2

    # 3. 최종 컨텍스트 포맷팅 테스트
    state = {"documents": docs, "input": "질문"}
    result = format_context(state)
    context_str = result["context"]

    print("\n[Test] 최종 컨텍스트 문자열 출력:")
    print("-" * 40)
    print(context_str)
    print("-" * 40)

    # 4. 검증: [p.X] 형식이 포함되어 있는가?
    assert "[p.1]" in context_str
    assert "[p.2]" in context_str
    assert context_str.startswith("[p.1]")

    # 페이지 1의 내용들이 하나로 합쳐져서 나타나는지 확인
    assert "첫 내용" in context_str
    assert "이어진 내용" in context_str

    print("\n✅ 메타데이터 전파 및 컨텍스트 포맷팅 테스트 성공!")


if __name__ == "__main__":
    test_metadata_to_context_string_conversion()
