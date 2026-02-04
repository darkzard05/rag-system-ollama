
import time
import numpy as np
import re
import sys
import os

def old_sequence_match(p_words_norm, key_seq):
    search_len = len(key_seq)
    best_start, max_match = -1, 0
    for i in range(len(p_words_norm) - search_len + 1):
        m_count = sum(1 for j in range(search_len) if p_words_norm[i + j] == key_seq[j])
        if m_count > max_match:
            max_match, best_start = m_count, i
            if max_match == search_len:
                break
    return best_start, max_match

def numpy_sequence_match(p_words_norm, key_seq):
    if not p_words_norm or not key_seq: return -1, 0
    
    # 단어들을 해시 정수로 변환하여 NumPy 배열 생성
    p_hashes = np.array([hash(w) for w in p_words_norm], dtype=np.int64)
    k_hashes = np.array([hash(w) for w in key_seq], dtype=np.int64)
    
    search_len = len(k_hashes)
    if len(p_hashes) < search_len: return -1, 0
    
    # 슬라이딩 윈도우 뷰 생성
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(p_hashes, search_len)
    
    # 모든 윈도우와 쿼리 시퀀스 비교 (Broadcast 비교)
    matches = (windows == k_hashes).sum(axis=1)
    
    best_idx = np.argmax(matches)
    return int(best_idx), int(matches[best_idx])

def run_match_benchmark():
    # 가상 데이터: 1,000개 단어가 있는 페이지, 12개 단어 검색
    p_words = [f"word_{i}" for i in range(1000)]
    key_seq = p_words[500:512] # 정답 포함
    
    # 무작위 노이즈 섞기
    for i in range(0, 1000, 10): p_words[i] = "noise"

    print(f"Benchmarking Sequence Matching (Page: 1000 words, Query: 12 words)")
    print("-" * 60)

    t0 = time.perf_counter()
    for _ in range(100):
        old_sequence_match(p_words, key_seq)
    t_old = (time.perf_counter() - t0) / 100 * 1000

    t1 = time.perf_counter()
    for _ in range(100):
        numpy_sequence_match(p_words, key_seq)
    t_new = (time.perf_counter() - t1) / 100 * 1000

    print(f"{'Method':<20} | {'Time (ms)':<12}")
    print("-" * 60)
    print(f"{'Old (Python Loop)':<20} | {t_old:10.4f}")
    print(f"{'New (NumPy Window)':<20} | {t_new:10.4f}")
    print(f"Speedup: {t_old/t_new:8.1f}x")

if __name__ == "__main__":
    run_match_benchmark()
