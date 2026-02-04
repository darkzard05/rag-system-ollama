
import sys
import time
import re
from pathlib import Path
import numpy as np

# 프로젝트 경로 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.model_loader import load_embedding_model

class KoreanCleaner:
    NUMBER_MAP = str.maketrans("0123456789", "영일이삼사오육칠팔구")
    
    @classmethod
    def clean(cls, text):
        if not text: return ""
        # 1. 공백 및 줄바꿈 정제
        text = re.sub(r'\s+', ' ', text).strip()
        # 2. 영어 한글화 (간이 버전)
        text = cls._normalize_english(text)
        # 3. 숫자 한글화
        text = text.translate(cls.NUMBER_MAP)
        return text

    @classmethod
    def _normalize_english(cls, text):
        upper_map = {"PDF": "피디에프", "AI": "에이아이", "RAG": "래그"}
        for k, v in upper_map.items():
            text = text.replace(k, v).replace(k.lower(), v)
        return text

def calculate_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def run_benchmark():
    print("=== [Benchmark] KoreanCleaner Effect Analysis ===")
    
    # 1. 모델 로드
    embedder = load_embedding_model("paraphrase-multilingual-MiniLM-L12-v2")
    
    # 2. 테스트 데이터 세트
    test_cases = [
        {
            "query": "2026년 프로젝트 계획",
            "doc": "이천이십육년 프로젝트 계획서 예안",
            "name": "Number Mismatch (2026 vs 이천이십육)"
        },
        {
            "query": "PDF 분석 기능",
            "doc": "문서 피디에프 파일을 정밀 분석하는 알고리즘",
            "name": "English Abbreviation (PDF vs 피디에프)"
        },
        {
            "query": "시스템 최적화 방법",
            "doc": "시  스템   최적화\n방법론에 대한 연구",
            "name": "Whitespace Noise (Multi-spaces/Newline)"
        }
    ]

    for case in test_cases:
        query = case["query"]
        doc = case["doc"]
        
        # [A] 원본 비교
        v_q_orig = embedder.embed_query(query)
        v_d_orig = embedder.embed_query(doc)
        sim_orig = calculate_similarity(v_q_orig, v_d_orig)
        
        # [B] 정규화 비교
        q_clean = KoreanCleaner.clean(query)
        d_clean = KoreanCleaner.clean(doc)
        v_q_clean = embedder.embed_query(q_clean)
        v_d_clean = embedder.embed_query(d_clean)
        sim_clean = calculate_similarity(v_q_clean, v_d_clean)
        
        improvement = (sim_clean - sim_orig) / sim_orig * 100
        
        print(f"\n[Case] {case['name']}")
        print(f"  - Original Score: {sim_orig:.4f}")
        print(f"  - Cleaned  Score: {sim_clean:.4f} ({improvement:+.2f}%)")
        print(f"  - Normalized Doc: {d_clean[:50]}...")

if __name__ == "__main__":
    run_benchmark()
