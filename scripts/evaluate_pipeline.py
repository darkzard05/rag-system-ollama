import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from src.core.rag_core import RAGSystem
from src.core.model_loader import ModelManager
from src.core.session import SessionManager
from src.api.streaming_handler import StreamingResponseHandler
from src.common.config import DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL
from src.common.logging_config import setup_logging


async def generate_evaluation_questions(rag, embedder, pdf_path):
    """
    평가 품질을 위해 검증된 고정 질문 셋을 사용합니다.
    """
    print("\n[1/4] Using expert-curated evaluation questions...")

    # 파이프라인 구축 (필수)
    file_name = os.path.basename(pdf_path)
    await rag.build_pipeline(pdf_path, file_name, embedder)

    questions = [
        "What is the primary objective of the CM3 model as described in the paper?",
        "How does the tokenization process for images work in the CM3 architecture?",
        "In what specific ways does CM3 differ from the DALL-E model?",
    ]

    print(f"   Loaded {len(questions)} expert questions.")
    return questions


async def run_evaluation():
    setup_logging(log_level="INFO")

    print("\n" + "=" * 50)
    print("[E2E] RAG Pipeline Evaluation Suite")
    print("=" * 50)

    # 1. 세션 및 RAG 초기화
    session_id = f"eval-session-{int(datetime.now().timestamp())}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    # 2. 모델 준비
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

    # 3. 질문 생성
    questions = await generate_evaluation_questions(rag, embedder, test_pdf)

    # 4. 질의 실행 및 결과 수집
    print("\n[2/4] Running evaluation queries...")
    results = []

    # 기존 결과 로드 (재시작 지원)
    if os.path.exists("eval_results.json"):
        try:
            with open("eval_results.json", "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"   Loaded {len(results)} existing results. Resuming...")
        except Exception:
            results = []

    for i, query in enumerate(questions):
        # 중복 실행 방지 (단순 쿼리 내용 기반)
        if any(r["query"] == query for r in results):
            print(f"   [{i + 1}/{len(questions)}] Skipping already processed query.")
            continue

        print(f"   [{i + 1}/{len(questions)}] Querying: '{query[:50]}...'")

        start_t = asyncio.get_event_loop().time()

        try:
            # 타임아웃 설정하여 일부 응답이라도 확보
            query_res = await asyncio.wait_for(
                rag.aquery(query, model_name=DEFAULT_OLLAMA_MODEL), timeout=100.0
            )

            full_response = query_res.get("response", "")
            context_docs = query_res.get("context", "")
            perf = query_res.get("performance", {})

        except asyncio.TimeoutError:
            print(f"   ⚠️ Query timed out after 60s")
            full_response = "TIMEOUT"
            context_docs = "TIMEOUT"
            perf = {}
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

        q_time = asyncio.get_event_loop().time() - start_t

        res_item = {
            "id": i + 1,
            "query": query,
            "response": full_response,
            "context": context_docs,
            "metrics": {
                "total_time": q_time,
                "ttft": perf.get("ttft"),
                "tps": perf.get("tps"),
                "input_tokens": perf.get("input_token_count"),
                "output_tokens": perf.get("token_count"),
            },
        }
        results.append(res_item)

        # 매 쿼리마다 저장
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n[3/4] Final results saved to eval_results.json.")
    print("\n[4/4] Evaluation completed successfully.")
    print(f"Total results: {len(results)} / {len(questions)}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_evaluation())
    except KeyboardInterrupt:
        print("\nEvaluation cancelled.")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
