import pytest
from common.utils import apply_tooltips_to_response
from langchain_core.documents import Document

def test_citation_with_section_name():
    """섹션명이 포함된 인용구 처리 및 툴팁 매칭 테스트"""
    # Setup
    response = "CM3 모델은 새로운 목적 함수를 사용합니다 [섹션: 3 CM3, p.3]."
    docs = [
        Document(
            page_content="이것은 1페이지 내용입니다.", 
            metadata={"page": 1, "current_section": "ABSTRACT"}
        ),
        Document(
            page_content="CM3의 핵심 원리는 마스킹입니다.", 
            metadata={"page": 3, "current_section": "3 CM3"}
        ),
        Document(
            page_content="3페이지의 다른 섹션 내용입니다.", 
            metadata={"page": 3, "current_section": "4 EXPERIMENTS"}
        )
    ]
    
    # Execute
    result = apply_tooltips_to_response(response, docs)
    
    # Verify
    # 1. 인용구가 span으로 변환되었는지 확인
    assert '<span class="citation-highlight"' in result
    # 2. 섹션명이 일치하는 문서의 내용이 툴팁에 들어갔는지 확인
    assert 'title="CM3의 핵심 원리는 마스킹입니다...."' in result
    # 3. data-page 속성이 올바른지 확인
    assert 'data-page="3"' in result
    # 4. 원본 텍스트가 유지되는지 확인
    assert "[섹션: 3 CM3, p.3]" in result

def test_citation_normalization():
    """다양한 인용 패턴(DOC, page, 괄호 등) 지원 여부 테스트"""
    docs = [Document(page_content="내용", metadata={"page": 5})]
    
    # 유형 1: [DOC 1, p.5]
    res1 = apply_tooltips_to_response("참조 [DOC 1, p.5]", docs)
    assert 'data-page="5"' in res1
    
    # 유형 2: (p.5)
    res2 = apply_tooltips_to_response("참조 (p.5)", docs)
    assert 'data-page="5"' in res2
    
    # 유형 3: [5]
    res3 = apply_tooltips_to_response("참조 [5]", docs)
    assert 'data-page="5"' in res3

def test_tooltip_escaping():
    """툴팁 내용의 HTML 이스케이프 및 줄바꿈 처리 테스트"""
    docs = [Document(page_content='그는 "안녕"이라고\n말했다.', metadata={"page": 10})]
    
    result = apply_tooltips_to_response("참조 [p.10]", docs)
    
    # 따옴표(&quot;)와 줄바꿈(공백) 처리 확인
    assert "&quot;안녕&quot;" in result
    assert "\n" not in result # 줄바꿈이 제거되어야 함
