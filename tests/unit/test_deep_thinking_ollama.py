
import pytest
from langchain_core.messages import AIMessageChunk
from src.core.custom_ollama import DeepThinkingChatOllama

def test_convert_chunk_with_content_blocks():
    """최신 표준인 content_blocks에서 사고 과정과 본문을 추출하는지 검증"""
    model = DeepThinkingChatOllama(model="deepseek-r1")
    
    # 1. 사고 과정 블록
    chunk_thought = AIMessageChunk(content="", content_blocks=[{"type": "reasoning", "reasoning": "이것은 사고 과정입니다."}])
    c, t = model._convert_chunk_to_thought_and_content(chunk_thought)
    assert c == ""
    assert t == "이것은 사고 과정입니다."
    
    # 2. 텍스트 블록
    chunk_text = AIMessageChunk(content="", content_blocks=[{"type": "text", "text": "이것은 본문입니다."}])
    c, t = model._convert_chunk_to_thought_and_content(chunk_text)
    assert c == "이것은 본문입니다."
    assert t == ""
    
    # 3. 혼합 (드문 경우지만 방어적 로직 확인)
    chunk_mixed = AIMessageChunk(content="", content_blocks=[
        {"type": "reasoning", "reasoning": "생각 중..."},
        {"type": "text", "text": "답변입니다."}
    ])
    c, t = model._convert_chunk_to_thought_and_content(chunk_mixed)
    assert c == "답변입니다."
    assert t == "생각 중..."

def test_convert_chunk_with_complex_content():
    """리스트 형태의 복합 콘텐츠에서 추출하는지 검증 (Anthropic 스타일 등)"""
    model = DeepThinkingChatOllama(model="deepseek-r1")
    
    chunk = AIMessageChunk(content=[
        {"type": "reasoning", "reasoning": "사고 과정"},
        {"type": "text", "text": "본문"}
    ])
    c, t = model._convert_chunk_to_thought_and_content(chunk)
    assert c == "본문"
    assert t == "사고 과정"

def test_convert_chunk_with_legacy_additional_kwargs():
    """레거시 additional_kwargs에서 추출하는지 검증"""
    model = DeepThinkingChatOllama(model="deepseek-r1")
    
    chunk = AIMessageChunk(content="본문입니다.", additional_kwargs={"thinking": "옛날 방식 사고 과정"})
    c, t = model._convert_chunk_to_thought_and_content(chunk)
    assert c == "본문입니다."
    assert t == "옛날 방식 사고 과정"

def test_convert_chunk_plain_string():
    """일반 문자열 콘텐츠 처리 검증"""
    model = DeepThinkingChatOllama(model="deepseek-r1")
    
    chunk = AIMessageChunk(content="일반 답변입니다.")
    c, t = model._convert_chunk_to_thought_and_content(chunk)
    assert c == "일반 답변입니다."
    assert t == ""
