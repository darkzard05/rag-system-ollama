
import re
import html

# 실제 로직 복사 (테스트용)
_RE_CITATION_BLOCK = re.compile(
    r"([\[\(])((?:[Pp](?:age)?\.?\s*\d+(?:[\s,]*))+)(([\]\)]))",
    re.IGNORECASE,
)
_RE_EXTRACT_PAGES = re.compile(r"(\d+)")

def test_citation_replacement(text):
    def replacement(match):
        inner_text = match.group(2)
        pages = _RE_EXTRACT_PAGES.findall(inner_text)
        badges = [f"[Badge p.{p}]" for p in pages]
        return "".join(badges)
    
    return _RE_CITATION_BLOCK.sub(replacement, text)

# 테스트 케이스
test_cases = [
    "[p.47, p.28, p.94, p.68]",
    "[p.5, 6, 7]",
    "(p.123)",
    "이 내용은 중요합니다 [p.1, p.2].",
    "복합 인용 테스트 [page 10, page 11]"
]

print("=== Citation Multi-Page Support Test ===")
for case in test_cases:
    result = test_citation_replacement(case)
    print(f"Original: {case}")
    print(f"Result  : {result}")
    print("-" * 30)
