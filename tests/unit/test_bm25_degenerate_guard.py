"""BM25 퇴화(빈-토큰) 코퍼스 가드 회귀 테스트.

버그 배경: 업로드된 PDF가 파싱·청킹은 통과하지만 그 텍스트가
``bm25_tokenizer`` 로 빈 토큰 목록을 만들면, ``rank_bm25`` 의 ``_calc_idf`` 가
``len(self.idf)==0`` 분모로 ``ZeroDivisionError`` 를 일으켜 파이프라인 전체가
크래시했다. ``has_bm25_tokens`` + ``create_bm25_retriever`` 의 None 폴백으로
벡터 전용 검색을 제공하도록 수정한다.
"""

from unittest.mock import patch

from langchain_core.documents import Document

from common.text_utils import bm25_tokenizer, has_bm25_tokens
from core.retriever_factory import create_bm25_retriever


def _docs(*contents: str) -> list[Document]:
    return [Document(page_content=c) for c in contents]


NORMAL_DOCS = _docs(
    "GraphRAG combines knowledge graphs with retrieval augmented generation",
    "Vector store caching prevents redundant indexing",
)


class TestHasBm25Tokens:
    def test_normal_corpus_returns_true(self) -> None:
        assert has_bm25_tokens([d.page_content for d in NORMAL_DOCS])

    def test_empty_token_corpus_returns_false(self) -> None:
        # bm25_tokenizer("") == []  → 코퍼스 전체가 빈 토큰 → False
        assert not has_bm25_tokens([""] * 3)
        assert not has_bm25_tokens(["   ", "", "   "])

    def test_partially_empty_corpus_still_true(self) -> None:
        # 일부 문서만 빈 토큰이어도 하나라도 유효 토큰이 있으면 True
        assert has_bm25_tokens(["", "GraphRAG retrieval"])


class TestCreateBm25Retriever:
    def test_normal_corpus_returns_bm25_retriever(self) -> None:
        retriever = create_bm25_retriever(NORMAL_DOCS)
        assert retriever is not None
        # 실제 BM25Retriever 객체가 생성되어야 한다.
        assert retriever.docs is not None

    def test_empty_token_corpus_returns_none(self) -> None:
        retriever = create_bm25_retriever(_docs(""))
        assert retriever is None

    @patch("langchain_community.retrievers.BM25Retriever.from_documents")
    def test_empty_token_corpus_skips_from_documents(self, mock_from_documents) -> None:
        """빈-토큰 코퍼스에서는 BM25 생성(0나눗셈 유발) 호출 자체를 건너뛴다."""
        retriever = create_bm25_retriever(_docs(""))
        assert retriever is None
        mock_from_documents.assert_not_called()

    def test_whitespace_only_corpus_returns_none(self) -> None:
        retriever = create_bm25_retriever(_docs("   ", " \t "))
        assert retriever is None

    def test_tokenizer_returns_empty_for_repro_input(self) -> None:
        # 가드가 의미를 가지려면 우선 토크나이저가 실제로 []를 반환해야 한다.
        assert bm25_tokenizer("") == []
