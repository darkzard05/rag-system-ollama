import asyncio
import sys
import time
import logging
from pathlib import Path
import numpy as np

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.query_optimizer import RAGQueryOptimizer
from core.model_loader import load_llm, load_embedding_model
from core.session import SessionManager
from common.config import DEFAULT_OLLAMA_MODEL

logging.basicConfig(level=logging.ERROR)

async def test_pure_llm(query, llm):
    start = time.time()
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    
    system_msg = "You are a query intent classifier. Output ONLY 'A', 'B', 'C', or 'D'."
    bound_llm = llm.bind(stop=["\n", " ", "."], options={"temperature": 0.0, "num_predict": 5})
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Classify intent: A:GREETING, B:FACTOID, C:RESEARCH, D:SUMMARY\nQuery: {query}\nResult:")
    ])
    chain = prompt | bound_llm | StrOutputParser()
    response = await chain.ainvoke({"query": query})
    intent_code = response.strip().upper()[0] if response else "B"
    intent_map = {"A": "GREETING", "B": "FACTOID", "C": "RESEARCH", "D": "SUMMARY"}
    return intent_map.get(intent_code, "FACTOID"), (time.time() - start) * 1000

async def run_comparison():
    print("=" * 70)
    print("      Intent Analysis Method Comparison: Pure LLM vs Hybrid")
    print("=" * 70)
    
    try:
        llm = load_llm(DEFAULT_OLLAMA_MODEL)
        embedder = load_embedding_model()
        SessionManager.set("embedder", embedder)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    test_cases = [
        ("안녕하세요", "GREETING"),
        ("안녕", "GREETING"),
        ("너는 누구니?", "GREETING"),
        ("이 문서의 제목이 뭐야?", "FACTOID"),
        ("저자가 누구인지 알려줘", "FACTOID"),
        ("상세히 분석해줘", "RESEARCH"),
        ("비교 분석이 필요해", "RESEARCH"),
        ("전체 내용 요약해줘", "SUMMARY"),
        ("주요 주제를 정리해줄래?", "SUMMARY"),
    ]

    results_llm = []
    results_hybrid = []

    print(f"\n{'Query':<30} | {'Method':<10} | {'Result':<10} | {'Latency':<10}")
    print("-" * 70)

    for query, expected in test_cases:
        # 1. Pure LLM Test
        actual_llm, lat_llm = await test_pure_llm(query, llm)
        results_llm.append({"lat": lat_llm, "acc": actual_llm == expected})
        print(f"{query[:30]:<30} | {'Pure LLM':<10} | {actual_llm:<10} | {lat_llm:>7.1f}ms")

        # 2. Hybrid Test
        RAGQueryOptimizer._intent_cache.clear()
        start_hybrid = time.time()
        actual_hybrid = await RAGQueryOptimizer.classify_intent(query, llm)
        lat_hybrid = (time.time() - start_hybrid) * 1000
        results_hybrid.append({"lat": lat_hybrid, "acc": actual_hybrid == expected})
        print(f"{'':<30} | {'Hybrid':<10} | {actual_hybrid:<10} | {lat_hybrid:>7.1f}ms")
        print("-" * 70)

    # 최종 통계
    avg_lat_llm = sum(r["lat"] for r in results_llm) / len(test_cases)
    avg_lat_hybrid = sum(r["lat"] for r in results_hybrid) / len(test_cases)
    acc_llm = sum(1 for r in results_llm if r["acc"]) / len(test_cases) * 100
    acc_hybrid = sum(1 for r in results_hybrid if r["acc"]) / len(test_cases) * 100
    
    skipped_llm = sum(1 for i in range(len(test_cases)) if results_hybrid[i]["lat"] < results_llm[i]["lat"] * 0.5)

    print("\n" + "=" * 70)
    print("                   Final Benchmark Results")
    print("=" * 70)
    print(f"Metric              | Pure LLM          | Hybrid (Semantic+LLM)")
    print("-" * 70)
    print(f"Avg Latency         | {avg_lat_llm:>10.1f}ms | {avg_lat_hybrid:>10.1f}ms")
    print(f"Accuracy            | {acc_llm:>10.1f}%  | {acc_hybrid:>10.1f}%")
    print(f"LLM Bypass Rate     | {'0.0%':>10}      | {skipped_llm/len(test_cases)*100:>10.1f}%")
    print("-" * 70)
    print(f"Speed Improvement   | {'Reference':>10}      | {(avg_lat_llm/avg_lat_hybrid):.2f}x Faster")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_comparison())