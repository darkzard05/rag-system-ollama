import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from src.core.rag_core import RAGSystem
from src.core.model_loader import ModelManager
from src.common.config import DEFAULT_OLLAMA_MODEL, EVAL_JUDGE_MODEL
from src.core.session import SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def judge_response(llm, query, response, context):
    """LLM을 사용하여 답변의 충실도(Faithfulness)와 관련성(Relevancy)을 평가합니다."""
    prompt = f"""
    당신은 RAG 시스템의 품질 평가관입니다. 제공된 <context>를 바탕으로 <response>가 <query>에 대해 얼마나 정확하고 관련성 있게 답변했는지 평가하십시오.

    <query>: {query}
    <context>: {context}
    <response>: {response}

    평가 기준:
    1. Faithfulness (충실도): 답변이 오직 제공된 <context>에만 근거하고 있는가? (할루시네이션 여부)
    2. Relevancy (관련성): 답변이 사용자의 질문에 직접적이고 유용한 정보를 제공하는가?

    출력 형식은 반드시 다음과 같은 JSON이어야 합니다:
    {{
        "faithfulness": <1-5점>,
        "relevancy": <1-5점>,
        "reason": "<간결한 평가 이유>"
    }}
    """

    try:
        # judge_model 사용 (ModelManager를 통해 로드)
        judge_llm = await ModelManager.get_llm(EVAL_JUDGE_MODEL)
        # 단순 invoke 사용 (스트리밍 불필요)
        res = await judge_llm.ainvoke([{"role": "user", "content": prompt}])
        content = res.content if hasattr(res, "content") else str(res)

        # JSON 추출 (간단한 파싱)
        import re

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.error(f"Judging error: {e}")

    return {"faithfulness": 0, "relevancy": 0, "reason": "Error during evaluation"}


async def main():
    # 1. Load Golden Set
    golden_set_path = ROOT_DIR / "tests/data/golden_set.json"
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    # 2. Initialize RAG System
    session_id = f"eval-simple-session-{int(datetime.now().timestamp())}"
    SessionManager.init_session(session_id=session_id)
    rag_system = RAGSystem(session_id=session_id)

    pdf_path = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    embedder = await ModelManager.get_embedder("nomic-embed-text-v2-moe")
    await rag_system.build_pipeline(pdf_path, "eval_doc", embedder)

    results = []
    logger.info(f"Running quality audit on {len(golden_set)} cases...")

    for item in golden_set:
        query = item["query"]
        logger.info(f"Evaluating: {query[:50]}...")

        response_text = ""
        retrieved_docs = []

        try:
            # astream_events API는 제거됨 → astream 사용
            # (("custom" | "messages" | "updates"), data) 튜플을 생성하는 비동기 제너레이터
            event_stream = await rag_system.astream(
                query, model_name=DEFAULT_OLLAMA_MODEL
            )
            async for kind, data in event_stream:
                if kind == "messages":
                    chunk = data.get("chunk")
                    content = getattr(chunk, "content", chunk) if chunk else ""
                    if content:
                        response_text += content
                elif kind == "updates":
                    if isinstance(data, dict) and "retrieve" in data:
                        upd = data["retrieve"]
                        if isinstance(upd, dict):
                            retrieved_docs = upd.get("relevant_docs", [])
        except Exception as e:
            logger.error(f"Query error: {e}")
            continue

        context_text = "\n".join([d.page_content for d in retrieved_docs])

        # LLM Judge 평가
        score = await judge_response(None, query, response_text, context_text)

        results.append(
            {
                "query": query,
                "faithfulness": score["faithfulness"],
                "relevancy": score["relevancy"],
                "reason": score["reason"],
            }
        )

    # 3. 결과 집계
    if not results:
        logger.error("No results collected.")
        return

    avg_f = sum(r["faithfulness"] for r in results) / len(results)
    avg_r = sum(r["relevancy"] for r in results) / len(results)

    print("\n" + "=" * 50)
    print("📊 RAG Quality Audit Summary")
    print("=" * 50)
    print(f"Total Cases: {len(results)}")
    print(f"Avg Faithfulness: {avg_f:.2f} / 5.0")
    print(f"Avg Relevancy:    {avg_r:.2f} / 5.0")
    print("-" * 50)
    for i, r in enumerate(results):
        print(
            f"Case {i + 1}: F={r['faithfulness']}, R={r['relevancy']} | {r['reason']}"
        )
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
