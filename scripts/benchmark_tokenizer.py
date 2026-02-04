import time
import re
import sys
import os

# 현재 프로젝트의 토크나이저 로직 복사
_RE_KOREAN_TOKEN = re.compile(r"[가-힣]{2,}|[a-zA-Z]{2,}|[0-9]+")
PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "에", "로", "서", "들")

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
                if len(stem) >= 2:
                    final_tokens.append(stem)
            if len(token) >= 3:
                for i in range(len(token) - 1):
                    final_tokens.append(token[i : i + 2])
    return final_tokens

# 최적화 제안 버전
def optimized_bm25_tokenizer(text: str) -> list[str]:
    if not text: return []
    tokens = _RE_KOREAN_TOKEN.findall(text.lower())
    if not tokens: return text.split()
    
    final_tokens = []
    for token in tokens:
        final_tokens.append(token)
        first_char = token[0]
        if '가' <= first_char <= '힣':
            t_len = len(token)
            if t_len > 2 and token.endswith(PARTICLES):
                stem = token[:-1]
                if len(stem) >= 2:
                    final_tokens.append(stem)
            if t_len >= 3:
                # extend와 list comprehension을 사용하여 append 루프 제거
                final_tokens.extend([token[i:i+2] for i in range(t_len - 1)])
    return final_tokens

def run_benchmark():
    sample_text = """
    인공지능 기술은 현대 사회의 다양한 분야에서 혁신적인 변화를 이끌어내고 있습니다. 
    특히 거대언어모델(LLM)의 발전은 자연어 처리 분야의 새로운 지평을 열었으며, 
    검색 증강 생성(RAG) 시스템은 모델의 환각 현상을 줄이고 정확한 정보를 제공하는 핵심 기술로 주목받고 있습니다.
    이러한 시스템을 구축하기 위해서는 효율적인 텍스트 분할과 정밀한 검색 알고리즘이 필수적입니다.
    """ * 500
    
    print(f"Test text length: {len(sample_text)} characters")
    
    res_orig = original_bm25_tokenizer(sample_text)
    res_opt = optimized_bm25_tokenizer(sample_text)
    print(f"Results Match: {res_orig == res_opt}")
    
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        original_bm25_tokenizer(sample_text)
    t_orig = (time.perf_counter() - t0) / iters * 1000
    
    t1 = time.perf_counter()
    for _ in range(iters):
        optimized_bm25_tokenizer(sample_text)
    t_opt = (time.perf_counter() - t1) / iters * 1000
    
    print(f"Original: {t_orig:.2f} ms")
    print(f"Optimized: {t_opt:.2f} ms")
    print(f"Speedup: {t_orig/t_opt:.2f}x")

if __name__ == "__main__":
    run_benchmark()