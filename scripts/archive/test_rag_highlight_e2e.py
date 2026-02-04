import os
import sys
import asyncio
from pathlib import Path
import fitz

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from core.session import SessionManager
from common.utils import get_pdf_annotations, sync_run

async def run_e2e_highlight_test():
    print("=== [E2E] RAG 답변 - 하이라이트 정밀도 통합 테스트 ===\n")
    
    # 세션 ID 고정 (컨텍스트 유실 방지)
    session_id = "test-e2e-session"
    SessionManager.set_session_id(session_id)
    SessionManager.init_session(session_id)
    
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    test_query = "What are the key components of the CM3 architecture?"
    
    # 1. 시스템 초기화 및 모델 준비
    print(f"1. 시스템 및 모델 준비: {os.path.basename(test_pdf)}")
    rag = RAGSystem(session_id=session_id)
    embedder = ModelManager.get_embedder()
    
    # 기본 모델 가져오기 (LLM)
    from common.config import DEFAULT_OLLAMA_MODEL
    llm = ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    
    # 문서 로드
    msg, cache_used = await rag.load_document(test_pdf, "2201.07520v1.pdf", embedder)
    print(f"   로드 결과: {msg} (Cache: {cache_used})")
    
    # 경로 강제 재설정 (테스트 환경 보장)
    SessionManager.set("pdf_file_path", test_pdf)

    # 2. 질문 던지기 및 답변 생성
    print(f"\n2. 질문 실행: '{test_query}'")
    result = await rag.aquery(test_query, llm=llm)
    
    # 결과 구조 전수 조사
    print(f"   그래프 출력 키: {list(result.keys())}")
    if "route_decision" in result:
        print(f"   결정된 의도: {result['route_decision']}")
    
    answer = result.get("response", "")
    retrieved_docs = result.get("documents", [])
    annotations = result.get("annotations", [])
    
    print(f"   답변 생성 완료 ({len(answer)}자)")
    print(f"   참고된 문서 조각: {len(retrieved_docs)}개")
    
    # [Emergency Check] 만약 문서가 없다면 리트리버 직접 호출 테스트
    if not retrieved_docs:
        print("\n⚠️ 그래프에서 문서를 찾지 못했습니다. 리트리버 직접 조회를 시도합니다.")
        faiss_ret = SessionManager.get("faiss_retriever")
        if faiss_ret:
            retrieved_docs = await faiss_ret.ainvoke(test_query)
            print(f"   리트리버 직접 조회 결과: {len(retrieved_docs)}개 발견")
        else:
            print("   ❌ 세션에서 리트리버를 찾을 수 없습니다.")

    # 3. 하이라이트 좌표 생성
    if retrieved_docs:
        print("\n3. 하이라이트 좌표 생성 및 검증")
        if not annotations:
            annotations = get_pdf_annotations(test_pdf, retrieved_docs[:5])
        
        print(f"   최종 하이라이트 박스: {len(annotations)}개")

        # 4. 정밀 대조 검증
        print("\n4. 정밀 대조 결과")
        print("-" * 80)
        with fitz.open(test_pdf) as doc:
            for i, anno in enumerate(annotations[:5]):
                page = doc[anno['page']-1]
                rect = fitz.Rect(anno['x'], anno['y'], anno['x'] + anno['width'], anno['y'] + anno['height'])
                real_text = page.get_text("text", clip=rect).strip().replace("\n", " ")
                
                # 매칭되는 원본 찾기
                orig_text = "N/A"
                doc_idx_str = anno['id'].split('_')[1]
                if doc_idx_str.isdigit():
                    idx = int(doc_idx_str)
                    if idx < len(retrieved_docs):
                        orig_text = retrieved_docs[idx].page_content[:100]

                print(f"[Box {i+1}] P.{anno['page']} | 원본: {orig_text[:50]}...")
                print(f"          | 실물: {real_text[:50]}...")
                is_match = any(w.lower() in real_text.lower() for w in orig_text.split()[:2] if len(w) > 2)
                print(f"          | 판정: {'✅ PASS' if is_match else '❌ FAIL'}")
                print("-" * 80)
    else:
        print("\n❌ 검색된 문서가 없어 하이라이트 테스트를 진행할 수 없습니다.")

    print("\n✨ E2E 통합 테스트 완료.")

if __name__ == "__main__":
    asyncio.run(run_e2e_highlight_test())
