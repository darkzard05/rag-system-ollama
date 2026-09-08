"""Highlight query cleaning — migrated from scripts/test_highlight_query_cleaning.py.

원본은 실제 ``tests/data/2201.07520v1.pdf``(fitz) 를 읽어 두 전처리 로직을
비교했습니다. PDF 판독 부분은 제거하고, 순수 문자열 처리 로직(``current_logic``/
``improved_logic``)만 추출해 마크다운/HTML 샘플에 대한 동작을 assert 합니다.
라이브 PDF/네트워크 의존성 없음.
"""

import re


def current_logic(content: str) -> list[str]:
    """현재 src/common/utils.py에 있는 로직 (스크립트에서 추출)."""
    # [개선] 마크다운 특수문자(#, *, ` 등)를 제거하여 실제 PDF 텍스트와 매칭율 향상
    clean_content = re.sub(r"[#*`_~\[\]()]", "", content).lower()

    # [최적화] 청크 전체를 하이라이트하기 위해 모든 문장을 검색 대상으로 설정
    sentences = [
        s.strip()
        for s in re.split(r"[.!?\n]", clean_content)
        if len(s.strip()) > 8  # 너무 짧은 검색어는 무시
    ]
    return sentences


def improved_logic(content: str) -> list[str]:
    """개선된 검색 쿼리 전처리 로직 (스크립트에서 추출)."""
    # 1. HTML 태그 제거 (예: <img src="...">)
    text = re.sub(r"<[^>]+>", " ", content)

    # 2. 마크다운 및 특수문자 제거 (기존보다 강화, 따옴표 포함)
    text = re.sub(r"[#*`_~\[\]()\"']", "", text)

    # 3. 연속 공백 제거 및 앞뒤 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    # 4. 소문자 변환
    text = text.lower()

    # 5. 문장 분리 (줄바꿈 포함)
    raw_sentences = re.split(r"[.!?\n]", text)

    sentences: list[str] = []
    for s in raw_sentences:
        s = s.strip()

        # [필터링 1] 최소 길이 상향 (8 -> 20): 너무 짧은 문장은 오탐지의 원인
        if len(s) < 20:
            continue

        # [필터링 2] 숫자나 특수문자로만 구성된 쓰레기 데이터 제거
        if re.match(r"^[\d\s\W]+$", s):
            continue

        # [필터링 3] 표/그림 캡션 등 불필요한 메타데이터 제거 (휴리스틱)
        if re.match(r"^(table|figure|fig\.|tab\.)\s*\d+", s):
            continue

        # [필터링 4] 참고문헌 패턴 (예: [1], (2020)) 등으로 시작하는 경우
        if re.match(r"^[\(\[]\s*\d+\s*[\)\]]", s):
            continue

        sentences.append(s)

    return sentences


class TestHighlightQueryCleaning:
    def test_current_logic_strips_markdown_headers_and_emphasis(self) -> None:
        content = (
            "## **Deep Learning**\n"
            "This is the introduction sentence of the paper. "
            "*Attention* is important."
        )
        sentences = current_logic(content)

        # 마크다운 기호가 제거된 소문자 문장이 검색어로 추출된다.
        assert "deep learning" in sentences
        assert "this is the introduction sentence of the paper" in sentences
        assert "attention is important" in sentences

    def test_improved_logic_removes_html_and_drops_short_queries(self) -> None:
        content = (
            "<img src='figure.png'> "
            "This sentence contains enough words to pass the filter. "
            "# Short one"
        )
        sentences = improved_logic(content)

        assert "this sentence contains enough words to pass the filter" in sentences
        # 20자 미만의 짧은 문장(Short one)은 오탐지 방지 필터로 제거된다.
        assert "short one" not in sentences

    def test_improved_logic_filters_figure_and_table_captions(self) -> None:
        content = (
            "Figure 3 shows the experimental results of our proposed method. "
            "The results section discusses the performance in detail. "
            "Table 2 summarizes the hyperparameters used."
        )
        sentences = improved_logic(content)

        assert "the results section discusses the performance in detail" in sentences
        assert not any(s.startswith("figure") for s in sentences)
        assert not any(s.startswith("table") for s in sentences)

    def test_improved_logic_filters_short_sentences(self) -> None:
        content = (
            "A tiny phrase here. This longer sentence has enough words to be kept."
        )
        sentences = improved_logic(content)

        assert "a tiny phrase here" not in sentences
        assert "this longer sentence has enough words to be kept" in sentences

    def test_improved_logic_is_stricter_than_current(self) -> None:
        content = (
            "<tag>Deep learning is a powerful technique used across many domains.</tag> "
            "# Short phrase"
        )
        current_out = current_logic(content)
        improved_out = improved_logic(content)

        # improved_logic은 HTML 제거 + 길이 임계값 상향으로 더 적은 문장을 남긴다.
        assert current_out, "current_logic should extract at least one sentence"
        assert len(improved_out) < len(current_out)
        assert "short phrase" not in improved_out
        assert any("short phrase" in s for s in current_out)
