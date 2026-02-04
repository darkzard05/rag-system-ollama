import os
import sys
import json
from pathlib import Path

# src 디렉토리를 경로에 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from common.utils import get_pdf_annotations
from core.session import SessionManager
from langchain_core.documents import Document

def simulate_context_to_highlight_flow():
    print("=== [Simulation] RAG 컨텍스트 -> 하이라이트 데이터 흐름 확인 ===\n")
    
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    # 세션 매니저에 가짜 경로 설정 (utils에서 참조할 수 있게 함)
    SessionManager.set("pdf_file_path", test_pdf)

    # 1. 'retrieve' 단계에서 나온 가상의 검색 결과들 (실제 논문 텍스트 조각)
    print("1단계: 검색된 문서 조각 (Retrieval Results)")
    simulated_docs = [
        Document(
            page_content="CM3: A CAUSAL MASKED MULTIMODAL MODEL OF THE INTERNET",
            metadata={"source": "2201.07520v1.pdf", "page": 1, "chunk_index": 0}
        ),
        Document(
            page_content="In this paper, we introduce CM3, a causal masked multimodal model",
            metadata={"source": "2201.07520v1.pdf", "page": 1, "chunk_index": 1}
        ),
        Document(
            page_content="Table 1 shows the performance comparison across different model sizes",
            metadata={"source": "2201.07520v1.pdf", "page": 3, "chunk_index": 45}
        )
    ] 
    
    for i, doc in enumerate(simulated_docs):
        print(f"  [{i+1}] P.{doc.metadata['page']} (Index {doc.metadata['chunk_index']}): {doc.page_content[:50]}...")

    # 2. 'format_context' 단계: 컨텍스트 구성 및 하이라이트 추출
    print("\n2단계: 컨텍스트 포맷팅 및 하이라이트 추출 (Formatting)")
    
    # 실제 graph_builder.py 로직 재현
    pdf_path = SessionManager.get("pdf_file_path")
    annotations = get_pdf_annotations(pdf_path, simulated_docs)
    
    # LLM용 컨텍스트 문자열 생성
    formatted_context = []
    for doc in simulated_docs:
        formatted_context.append(f"-- PAGE {doc.metadata['page']} --\n{doc.page_content}")
    context_str = "\n\n".join(formatted_context)

    print(f"  - 생성된 하이라이트 개수: {len(annotations)}")
    print(f"  - LLM 주입 컨텍스트 길이: {len(context_str)}자")

    # 3. 최종 데이터 구조 (UI로 전달될 형태) 확인
    print("\n3단계: 최종 데이터 구조 (JSON Payload)")
    
    # 첫 3개만 샘플 출력
    for i, anno in enumerate(annotations[:3]):
        print(f"  하이라이트 {i+1}:")
        print(json.dumps(anno, indent=4))

    print("\n" + "="*50)
    print("✨ 분석 완료: 위 JSON 데이터가 UI의 pdf_viewer로 전달되어 색칠됩니다.")
    print("="*50)

if __name__ == "__main__":
    simulate_context_to_highlight_flow()
