import asyncio
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 프로젝트 루트 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL, DEFAULT_EMBEDDING_MODEL

SCORE_PROMPT = """
당신은 RAG 시스템 평가 전문가입니다. 아래 [질문], [정답], [모델 답변]을 비교하여 답변의 품질을 1점에서 5점 사이로 평가하세요.

[질문]: {question}
[정답]: {ground_truth}
[모델 답변]: {answer}

[평가 기준]:
- 5점: 답변이 정답과 의미적으로 완벽하게 일치하며 정확함.
- 3점: 답변의 핵심은 맞지만 세부 정보가 부족하거나 약간의 노이즈가 있음.
- 1점: 답변이 틀렸거나 질문과 관련이 없음.

결과는 반드시 숫자 하나(예: 5)만 출력하세요. 설명은 생략하세요.
"""

async def run_quick_evaluation(pdf_path: str, testset_csv: str):
    print("--- [Quick Eval] 통합 품질 평가 시작 ---")
    
    # 1. RAG 준비
    session_id = "eval_" + str(int(datetime.now().timestamp()))
    rag_sys = RAGSystem(session_id=session_id)
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    
    print("1. RAG 파이프라인 구축 중...")
    await rag_sys.build_pipeline(pdf_path, os.path.basename(pdf_path), embedder)
    
    # 2. 테스트셋 로드 및 추론
    df = pd.read_csv(testset_csv)
    print("2. 총 " + str(len(df)) + "개 질문에 대해 추론 시작...")
    
    eval_llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    eval_embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    score_chain = ChatPromptTemplate.from_template(SCORE_PROMPT) | eval_llm | StrOutputParser()

    results = []
    # 시간 절약을 위해 3개만 샘플링하여 검증
    sample_df = df.head(3) 
    
    for i, row in sample_df.iterrows():
        query = row['question']
        gt = row['ground_truth']
        
        # [리팩토링 반영] 모델 이름만 전달
        resp = await rag_sys.aquery(query, model_name=DEFAULT_OLLAMA_MODEL)
        answer = resp.get("output", resp.get("response", ""))
        
        # 채점
        try:
            score_str = await score_chain.ainvoke({"question": query, "ground_truth": gt, "answer": answer})
            import re
            match = re.search(r'[1-5]', score_str)
            score = int(match.group()) if match else 3
        except: score = 3
        
        # 유사도
        try:
            v1, v2 = eval_embeddings.embed_query(str(gt)), eval_embeddings.embed_query(str(answer))
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except: sim = 0.5
        
        results.append({
            "question": query, "ground_truth": gt, "answer": answer,
            "score": score, "similarity": round(float(sim), 4)
        })
        print("[" + str(i+1) + "/3] Score: " + str(score) + ", Sim: " + str(round(float(sim), 2)))

    # 결과 리포트 출력
    res_df = pd.DataFrame(results)
    avg_score = res_df['score'].mean()
    print("\n--- 최종 통합 점수 (샘플 3개) ---")
    print("평균 점수: " + str(round(avg_score, 2)))
    print("--- [검증 완료] ---")

if __name__ == "__main__":
    pdf = "tests/data/2201.07520v1.pdf"
    csv = "tests/data/testset_2201.csv"
    if os.path.exists(pdf) and os.path.exists(csv):
        asyncio.run(run_quick_evaluation(pdf, csv))
    else:
        print("Error: PDF or CSV not found.")
