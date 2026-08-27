from langchain_core.documents import Document
from src.core.graph_builder import _merge_adjacent_chunks, format_context


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

    # 2. 청크 병합 테스트
    merged_docs = _merge_adjacent_chunks(docs)

    assert len(merged_docs) == 2
    assert "이어진 내용" in merged_docs[0].page_content
    assert merged_docs[0].metadata["page"] == 1
    assert merged_docs[1].metadata["page"] == 2

    # 3. 최종 컨텍스트 포맷팅 테스트 (병합된 문서 기준)
    context_str = format_context(merged_docs)

    # 4. 검증: 리팩터 이후 컨텍스트 포맷은
    #    "[doc:<hash>] [section:일반 본문] [page:1] [score:0.000]" 형식이다.
    #    (기본 섹션 = "일반 본문", 페이지 메타데이터는 [page:N] 태그로 직렬화)
    assert "[section:일반 본문]" in context_str
    assert "[page:1]" in context_str
    assert "[page:2]" in context_str
    assert context_str.startswith("[doc:")

    # 페이지 1의 내용들이 하나로 합쳐져서 나타나는지 확인
    assert "첫 내용" in context_str
    assert "이어진 내용" in context_str
