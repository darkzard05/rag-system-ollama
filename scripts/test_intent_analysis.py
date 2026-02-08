import asyncio
import sys
from pathlib import Path
import logging

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.query_optimizer import RAGQueryOptimizer
from core.model_loader import load_llm, load_embedding_model
from core.session import SessionManager
from common.config import DEFAULT_OLLAMA_MODEL

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_intent_analysis():
    print("=" * 60)
    print("      Intent Analysis Comprehensive Test")
    print("=" * 60)
    
    # 모델 로드
    print("1. Loading Models...")
    try:
        llm = load_llm(DEFAULT_OLLAMA_MODEL)
        embedder = load_embedding_model()
        SessionManager.set("embedder", embedder)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 테스트 케이스 정의 (의도별 예상 결과)
    test_cases = [
        # GREETING
        ("안녕하세요!", "GREETING"),
        ("안녕", "GREETING"),
        ("너는 누구니?", "GREETING"),
        
        # FACTOID
        ("이 문서의 제목이 뭐야?", "FACTOID"),
        ("저자가 누구인지 알려줘", "FACTOID"),
        ("2023년 매출액이 얼마야?", "FACTOID"),
        
        # RESEARCH
        ("이 기술의 장점과 단점을 비교 분석해줘", "RESEARCH"),
        ("CM3 모델이 왜 성능이 좋은지 상세히 설명해줄래?", "RESEARCH"),
        ("데이터 편향성 문제와 모델 크기 사이의 상관관계가 뭐야?", "RESEARCH"),
        
        # SUMMARY
        ("문서 전체 내용을 3문장으로 요약해줘", "SUMMARY"),
        ("이 파일의 핵심 주제를 정리해줘", "SUMMARY"),
        ("내용을 질문과 답변 형식으로 재구성해줘", "SUMMARY"),
    ]

    results = []
    
    print("\n2. Starting Intent Classification...")
    print("-" * 60)
    print(f"{'Query':<40} | {'Expected':<10} | {'Actual':<10} | {'Status'}")
    print("-" * 60)

    for query, expected in test_cases:
        try:
            actual = await RAGQueryOptimizer.classify_intent(query, llm)
            status = "✅ PASS" if actual == expected else "❌ FAIL"
            print(f"{query[:40]:<40} | {expected:<10} | {actual:<10} | {status}")
            results.append({
                "query": query,
                "expected": expected,
                "actual": actual,
                "status": status
            })
        except Exception as e:
            print(f"Error testing query '{query}': {e}")

    # 결과 분석
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "✅ PASS")
    accuracy = (passed / total) * 100

    print("\n" + "=" * 60)
    print("               Final Evaluation")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {total - passed}")
    print(f"Accuracy   : {accuracy:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_intent_analysis())