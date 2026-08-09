import asyncio
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ruff: noqa: E402 - sys.path 부트스트랩 이후 임포트 (scripts 표준 패턴)
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL
from core.model_loader import ModelManager
from core.rag_core import RAGSystem

logger = logging.getLogger(__name__)

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

    # [R4-07 수정] 직접 ChatOllama/OllamaEmbeddings 생성 대신 ModelManager 경유
    # (config.yml temperature/num_ctx/num_predict 준수, 리소스 풀 공유)
    eval_llm = await ModelManager.get_llm(DEFAULT_OLLAMA_MODEL, temperature=0.0)
    eval_embeddings = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    score_chain = (
        ChatPromptTemplate.from_template(SCORE_PROMPT) | eval_llm | StrOutputParser()
    )

    results = []
    # 시간 절약을 위해 3개만 샘플링하여 검증
    sample_df = df.head(3)
    total = len(sample_df)

    for seq, (_, row) in enumerate(sample_df.iterrows(), start=1):
        query = row["question"]
        gt = row["ground_truth"]

        # [리팩토링 반영] 모델 이름만 전달
        resp = await rag_sys.aquery(query, model_name=DEFAULT_OLLAMA_MODEL)
        answer = resp.get("output", resp.get("response", ""))

        # 채점 (R4-07: bare except 기본값 대입 금지 - 실패 시 해당 샘플 누락 표기)
        score = None
        score_error = None
        try:
            score_str = await score_chain.ainvoke(
                {"question": query, "ground_truth": gt, "answer": answer}
            )
            match = re.search(r"[1-5]", score_str)
            if match is None:
                raise ValueError("judge 응답에서 1-5 점수 미발견")
            score = int(match.group())
        except Exception as exc:  # noqa: BLE001 - 호출/파싱 실패는 샘플 스킵
            score_error = "judge 측정 실패: " + str(exc)
            logger.warning("[Quick Eval] 샘플 %s score 측정 건너뜀: %s", seq, exc)

        # 유사도 (R4-07: bare except 기본값 대입 금지 - 실패 시 해당 샘플 누락 표기)
        sim = None
        sim_error = None
        try:
            v1, v2 = (
                eval_embeddings.embed_query(str(gt)),
                eval_embeddings.embed_query(str(answer)),
            )
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        except Exception as exc:  # noqa: BLE001 - 임베딩/수치 실패는 샘플 스킵
            sim_error = "similarity 측정 실패: " + str(exc)
            logger.warning("[Quick Eval] 샘플 %s similarity 측정 건너뜀: %s", seq, exc)

        results.append(
            {
                "question": query,
                "ground_truth": gt,
                "answer": answer,
                "score": score,
                "similarity": round(sim, 4) if sim is not None else None,
                "score_error": score_error,
                "sim_error": sim_error,
            }
        )
        score_show = str(round(score, 2)) if score is not None else "N/A"
        sim_show = str(round(sim, 2)) if sim is not None else "N/A"
        print(
            "["
            + str(seq)
            + "/"
            + str(total)
            + "] Score: "
            + score_show
            + ", Sim: "
            + sim_show
        )

    # 결과 리포트 출력 (누락 샘플 표기)
    res_df = pd.DataFrame(results)
    valid_scores = res_df["score"].dropna()
    valid_sims = res_df["similarity"].dropna()
    avg_score = valid_scores.mean()
    avg_sim = valid_sims.mean()
    print("\n--- 최종 통합 점수 (샘플 " + str(total) + "개) ---")
    print(
        "평균 점수: "
        + (str(round(float(avg_score), 2)) if len(valid_scores) else "N/A")
        + " (측정 "
        + str(len(valid_scores))
        + "/"
        + str(total)
        + "건)"
    )
    print(
        "평균 유사도: "
        + (str(round(float(avg_sim), 2)) if len(valid_sims) else "N/A")
        + " (측정 "
        + str(len(valid_sims))
        + "/"
        + str(total)
        + "건)"
    )
    for r in results:
        if r["score_error"] or r["sim_error"]:
            print(
                "[누락] "
                + str(r["question"])[:40]
                + ": "
                + (r["score_error"] or r["sim_error"])
            )
    print("--- [검증 완료] ---")


if __name__ == "__main__":
    pdf = "tests/data/2201.07520v1.pdf"
    csv = "tests/data/testset_2201.csv"
    if os.path.exists(pdf) and os.path.exists(csv):
        asyncio.run(run_quick_evaluation(pdf, csv))
    else:
        print("Error: PDF or CSV not found.")
