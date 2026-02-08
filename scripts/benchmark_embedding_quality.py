
import numpy as np
import logging
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.model_loader import load_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def evaluate_embeddings():
    print("=" * 60)
    print("      Embedding Model (nomic-embed-text) Quality Audit")
    print("=" * 60)

    try:
        embedder = load_embedding_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    test_groups = [
        {
            "name": "1. Paraphrase (Same Meaning, Different Words)",
            "pairs": [
                ("이 문서를 요약해줘", "이 파일의 내용을 짧게 줄여줄래?"),
                ("What is the main idea?", "Tell me the core concept.")
            ]
        },
        {
            "name": "2. High Confusion (Different Intent, Similar Context)",
            "pairs": [
                ("이 내용을 요약해줘", "이 내용을 분석해줘"),  # Summary vs Research
                ("저자가 누구야?", "이 문서의 제목이 뭐야?")  # Factoid vs Factoid
            ]
        },
        {
            "name": "3. Cross-lingual (Korean vs English)",
            "pairs": [
                ("이 문서를 요약해줘", "Summarize this document"),
                ("인공지능의 미래", "The future of Artificial Intelligence")
            ]
        },
        {
            "name": "4. Irrelevant (Random Noise)",
            "pairs": [
                ("오늘 날씨 어때?", "이 문서의 결론이 뭐야?"),
                ("Apple is a fruit", "The model achieved 90% accuracy")
            ]
        }
    ]

    for group in test_groups:
        print(f"\n{group['name']}")
        print("-" * 60)
        for s1, s2 in group['pairs']:
            v1 = embedder.embed_query(s1)
            v2 = embedder.embed_query(s2)
            sim = cosine_similarity(v1, v2)
            print(f"S1: {s1}")
            print(f"S2: {s2}")
            print(f"Score: {sim:.4f}\n")

if __name__ == "__main__":
    evaluate_embeddings()
