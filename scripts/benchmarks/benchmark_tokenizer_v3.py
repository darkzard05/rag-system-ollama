
import time
import re
import sys
import os

_RE_KOREAN_TOKEN = re.compile(r"[가-힣]{2,}|[a-zA-Z]{2,}|[0-9]+")
PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "에", "로", "서", "들")
# 조사 제거용 정규식 (어근 2자 이상 + 조사 + 한글이 아닌 경계)
STEM_PATTERN = re.compile(r'([가-힣]{2,})[은는이가을를의에로서들](?![가-힣])')
# Bi-gram 추출용 정규식 (한글+한글 조합만 추출, 단어 경계 자동 준수)
BIGRAM_PATTERN = re.compile(r'([가-힣])(?=([가-힣]))')

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

def multi_pass_regex_tokenizer(text: str) -> list[str]:
    if not text: return []
    lower_text = text.lower()
    
    # Pass 1: 기본 토큰 (한/영/숫자)
    tokens = _RE_KOREAN_TOKEN.findall(lower_text)
    
    # Pass 2: Bi-gram (전체 텍스트에서 한 번에 추출)
    # findall은 캡처 그룹이 있으면 튜플을 반환하므로 결합 필요
    bigrams = [m[0] + m[1] for m in BIGRAM_PATTERN.findall(lower_text)]
    
    # Pass 3: 조사 제거 어근
    stems = STEM_PATTERN.findall(lower_text)
    
    # 모든 결과 통합 (Python 루프 없음)
    return tokens + bigrams + stems

def run_benchmark():
    sample_text = """
    인공지능 기술은 현대 사회의 다양한 분야에서 혁신적인 변화를 이끌어내고 있습니다. 
    특히 거대언어모델(LLM)의 발전은 자연어 처리 분야의 새로운 지평을 열었으며, 
    검색 증강 생성(RAG) 시스템은 모델의 환각 현상을 줄이고 정확한 정보를 제공하는 핵심 기술로 주목받고 있습니다.
    이러한 시스템을 구축하기 위해서는 효율적인 텍스트 분할과 정밀한 검색 알고리즘이 필수적입니다.
    """ * 1000
    
    print(f"Test text length: {len(sample_text)} characters")
    
    # 결과가 완벽히 일치하지는 않을 수 있음 (조사 제거 로직의 정교함 차이)
    # 하지만 BM25의 목적(키워드 추출)에는 부합하는지 확인
    res_orig = original_bm25_tokenizer(sample_text)
    res_multi = multi_pass_regex_tokenizer(sample_text)
    
    print(f"Original token count: {len(res_orig)}")
    print(f"Multi-pass token count: {len(res_multi)}")
    
    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters): original_bm25_tokenizer(sample_text)
    t_orig = (time.perf_counter() - t0) / iters * 1000
    
    t1 = time.perf_counter()
    for _ in range(iters): multi_pass_regex_tokenizer(sample_text)
    t_multi = (time.perf_counter() - t1) / iters * 1000
    
    print(f"Original: {t_orig:.2f} ms")
    print(f"Multi-pass Regex: {t_multi:.2f} ms")
    print(f"Speedup: {t_orig/t_multi:.2f}x")

if __name__ == "__main__":
    run_benchmark()
