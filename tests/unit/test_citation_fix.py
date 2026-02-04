import re
import pytest

# 실제 로직 복사 (src/common/utils.py 와 일치시켜야 함)
_RE_CITATION_BLOCK = re.compile(
    r"([\[\(])((?:[Pp](?:age)?\.?\s*)?\d+(?:[\s,]*)(?:(?:[Pp](?:age)?\.?\s*)?\d+(?:[\s,]*))*)([\]\)])",
    re.IGNORECASE,
)
_RE_EXTRACT_PAGES = re.compile(r"(\d+)")

def process_citation_replacement(text):
    def replacement(match):
        inner_text = match.group(2)
        pages = _RE_EXTRACT_PAGES.findall(inner_text)
        badges = [f"[Badge p.{p}]" for p in pages]
        return "".join(badges)
    
    return _RE_CITATION_BLOCK.sub(replacement, text)

@pytest.mark.parametrize("input_text, expected", [
    ("[p.47, p.28, p.94, p.68]", "[Badge p.47][Badge p.28][Badge p.94][Badge p.68]"),
    ("[p.5, 6, 7]", "[Badge p.5][Badge p.6][Badge p.7]"),
    ("(p.123)", "[Badge p.123]"),
    ("이 내용은 중요합니다 [p.1, p.2].", "이 내용은 중요합니다 [Badge p.1][Badge p.2]."),
    ("복합 인용 테스트 [page 10, page 11]", "복합 인용 테스트 [Badge p.10][Badge p.11]")
])
def test_citation_replacement(input_text, expected):
    result = process_citation_replacement(input_text)
    assert result == expected
