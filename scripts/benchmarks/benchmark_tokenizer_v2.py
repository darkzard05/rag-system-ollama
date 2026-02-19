
import time
import re
import sys
import os

_RE_KOREAN_TOKEN = re.compile(r"[가-힣]{2,}|[a-zA-Z]{2,}|[0-9]+")
PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "에", "로", "서", "들")
BIGRAM_EXTRACTOR = re.compile(r'(?=([가-힣]{2}))')

def original_bm25_tokenizer(text: str) -> list[str]:
    if not text: return []
    tokens = _RE_KOREAN_TOKEN.findall(text.lower())
    if not tokens: return text.split()
    final_tokens = []
    for token in tokens:
        final_tokens.append(token)
        if "가" <= token[0] <= "힣":
            if len(token) > 2 and token.endswith(PARTICLES):
                stem = token[:-1]
                if len(stem) >= 2: final_tokens.append(stem)
            if len(token) >= 3:
                for i in range(len(token) - 1):
                    final_tokens.append(token[i : i + 2])
    return final_tokens

def regex_optimized_tokenizer(text: str) -> list[str]:
    if not text: return []
    # 1. 텍스트 소문자화 및 기본 토큰 추출
    tokens = _RE_KOREAN_TOKEN.findall(text.lower())
    if not tokens: return text.split()
    
    final_tokens = []
    # 리스트 append 대신 미리 필요한 크기를 짐작할 수 없으므로 append 사용하되 루프 최소화
    for token in tokens:
        final_tokens.append(token)
        if '가' <= token[0] <= '힣':
            # 조사 제거
            if len(token) > 2 and token.endswith(PARTICLES):
                stem = token[:-1]
                if len(stem) >= 2: final_tokens.append(stem)
            
            # Bi-gram: 정규식 룩어헤드로 추출 (C-level 루프 활용)
            if len(token) >= 3:
                final_tokens.extend(BIGRAM_EXTRACTOR.findall(token))
                
    return final_tokens

def run_benchmark():
    sample_text = """
    인공지능 기술은 현대 사회의 다양한 분야에서 혁신적인 변화를 이끌어내고 있습니다. 
    특히 거대언어모델(LLM)의 발전은 자연어 처리 분야의 새로운 지평을 열었으며, 
    검색 증강 생성(RAG) 시스템은 모델의 환각 현상을 줄이고 정확한 정보를 제공하는 핵심 기술로 주목받고 있습니다.
    이러한 시스템을 구축하기 위해서는 효율적인 텍스트 분할과 정밀한 검색 알고리즘이 필수적입니다.
    """ * 1000
    
    print(f"Test text length: {len(sample_text)} characters")
    
    res_orig = original_bm25_tokenizer(sample_text)
    res_reg = regex_optimized_tokenizer(sample_text)
    print(f"Results Match: {res_orig == res_reg}")
    
    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters): original_bm25_tokenizer(sample_text)
    t_orig = (time.perf_counter() - t0) / iters * 1000
    
    t1 = time.perf_counter()
    for _ in range(iters): regex_optimized_tokenizer(sample_text)
    t_reg = (time.perf_counter() - t1) / iters * 1000
    
    print(f"Original: {t_orig:.2f} ms")
    print(f"Regex Optimized: {t_reg:.2f} ms")
    print(f"Speedup: {t_orig/t_reg:.2f}x")

if __name__ == "__main__":
    run_benchmark()
