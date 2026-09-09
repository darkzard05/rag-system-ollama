
import asyncio
import time
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL

async def run_e2e_benchmark():
    pdf_path = "tests/data/2201.07520v1.pdf"
    file_name = os.path.basename(pdf_path)
    
    print("\n" + "="*60)
    print(f"🚀 [E2E 통합 벤치마크 시작] 파일: {file_name}")
    print("="*60)

    rag_sys = RAGSystem(session_id="benchmark_session")
    embedder = await ModelManager.get_embedder()

    # 1. 인덱싱 단계 성능 측정
    start_time = time.time()
    print("\n[Step 1] 문서 파싱 및 인덱싱 중...")
    msg, cache_used = await rag_sys.build_pipeline(pdf_path, file_name, embedder)
    indexing_time = time.time() - start_time
    print(f"✅ 인덱싱 완료 ({indexing_time:.2f}s) | 캐시 사용: {cache_used}")

    # 2. 질의응답 단계 성능 측정 (복합 질문)
    query = "CM3 모델의 'Causal Masking' 목표(Objective)에 대해 Methodology 섹션의 내용을 바탕으로 설명해주고 정확한 출처를 표기해줘."
    print(f"\n[Step 2] 질의 실행: '{query}'")
    
    start_time = time.time()
    result = await rag_sys.aquery(query)
    total_latency = time.time() - start_time

    # 3. 결과 분석 및 지표 출력
    print("\n" + "-"*60)
    print(f"📊 [벤치마크 지표]")
    print(f"- 총 응답 시간: {total_latency:.2f}s")
    print(f"- 검색 결과 수: {len(result['documents'])}개")
    if 'performance' in result:
        perf = result['performance']
        gen_time = perf.get('total_duration', 0) / 1e9 # ns to s
        print(f"- 생성 소요 시간: {gen_time:.2f}s")
        print(f"- 초당 토큰 생성(TPS): {perf.get('eval_count', 0) / (gen_time or 1):.2f}")
    print("-"*60)

    print("\n[Step 3] 최종 답변 및 인용 확인:")
    print("\n" + result['response'])
    print("\n" + "="*60)

    # 섹션 인용 여부 자동 검사
    response = result['response']
    if "[" in response and "p." in response:
        print("✅ 검증 성공: 섹션 기반 인용 형식이 답변에 포함되었습니다.")
    else:
        print("⚠️ 검증 주의: 인용 형식이 예상과 다를 수 있습니다. 답변 본문을 확인하십시오.")

if __name__ == "__main__":
    asyncio.run(run_e2e_benchmark())
