import os
import sys
import asyncio
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


async def run_full_pipeline_test():
    # [동기화] 실제 앱과 동일한 로깅 설정 적용
    setup_logging(log_level="INFO")

    print("\n" + "=" * 50)
    print("[E2E] RAG Pipeline Integration Test (App Synchronized)")
    print("설명: 실제 앱(UI/API)과 동일한 호출 방식으로 파이프라인을 검증합니다.")
    print("=" * 50)

    # 1. 세션 초기화 (실제 앱과 동일한 라이프사이클)
    session_id = f"test-session-{int(datetime.now().timestamp())}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    # 2. 모델 준비
    print("\n1. Preparing Embedding Model...")
    try:
        # 임베더는 인덱싱을 위해 명시적으로 필요
        embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 3. 문서 인덱싱
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = os.path.basename(test_pdf)
    print(f"\n2. Indexing Document: {file_name}")

    start_time = asyncio.get_event_loop().time()
    # build_pipeline 내부에서 ResourcePool 등록 및 세션 정보 저장이 일어남
    msg, cache_used = await rag.build_pipeline(test_pdf, file_name, embedder)
    load_time = asyncio.get_event_loop().time() - start_time
    print(f"   Result: {msg} (Cache: {cache_used}) | Time: {load_time:.2f}s")

    # 4. 통합 인터페이스 질의 (실제 앱 호출 방식과 100% 일치)
    test_cases = [
        "CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤 차이가 있어?"
    ]

    print("\n3. Running App-Synchronized Queries (Streaming Mode)...")
    for i, test_query in enumerate(test_cases):
        print(f"   [{i + 1}/{len(test_cases)}] Querying: '{test_query[:50]}...'")

        # UI와 동일한 스트리밍 파이프라인 구축
        handler = StreamingResponseHandler()
        full_response = ""
        final_perf = {}

        start_t = asyncio.get_event_loop().time()

        try:
            # 1. RAG 엔진으로부터 이벤트 스트림 생성
            event_stream = await rag.astream(test_query, model_name=DEFAULT_OLLAMA_MODEL)

            # 2. 핸들러를 통해 이벤트를 소비하며 지표 계산 및 청크 생성
            async for chunk in handler.stream_graph_events(event_stream):
                if chunk.content:
                    full_response += chunk.content

                if chunk.performance:
                    # 마지막 performance 데이터를 최종 지표로 저장
                    final_perf = chunk.performance

                if chunk.is_final:
                    # 최종 chunk의 performance가 가장 정확함
                    final_perf = chunk.performance or final_perf

        except Exception as e:
            print(f"   ❌ Streaming Error: {e}")
            continue

        q_time = asyncio.get_event_loop().time() - start_t

        # 지표 추출
        ttft = final_perf.get("ttft")
        tps = final_perf.get("tps")
        in_tok = final_perf.get("input_token_count")
        out_tok = final_perf.get("token_count")

        ttft_str = f"{ttft:.3f}s" if ttft is not None else "N/A"
        tps_str = f"{tps:.2f} t/s" if tps is not None else "N/A"
        in_tok_str = str(in_tok) if in_tok is not None else "N/A"
        out_tok_str = str(out_tok) if out_tok is not None else "N/A"

        print(f"   -> Done ({q_time:.2f}s)")
        print(f"   -> Response Length: {len(full_response)} chars")
        print(
            f"   -> Metrics: TTFT={ttft_str}, TPS={tps_str}, Tokens={in_tok_str}/{out_tok_str}"
        )

        # 기능적 및 성능적 검증
        perf_ok = (
            (tps is not None and tps > 0 and out_tok is not None and out_tok > 0)
            if full_response
            else True
        )
        if len(full_response) > 100 and perf_ok:
            print(
                f"   ✅ [PASS] Query {i + 1} functional & performance check successful."
            )
        else:
            print(
                f"   ❌ [FAIL] Query {i + 1} produced suspicious output or invalid metrics."
            )
            if not perf_ok:
                print(f"      (Reason: TPS={tps}, Tokens={out_tok})")

    print(f"\n4. Pipeline Test Finished (Total Queries: {len(test_cases)})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_full_pipeline_test())
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
