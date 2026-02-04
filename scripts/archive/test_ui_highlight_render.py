import os
import sys
import streamlit as st
from pathlib import Path

# src 디렉토리를 경로에 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from streamlit_pdf_viewer import pdf_viewer
from common.utils import get_pdf_annotations
from core.session import SessionManager

st.set_page_config(layout="wide")

def test_ui_render():
    st.title("🧪 PDF 하이라이트 렌더링 독립 테스트")
    st.write("이 테스트는 RAG 로직 없이 UI 렌더링 성능과 하이라이트 기능을 직접 검증합니다.")

    # 1. 테스트용 PDF 찾기
    pdf_dir = ROOT_DIR / "data" / "temp"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        st.error("❌ data/temp 디렉토리에 PDF 파일이 없습니다.")
        return

    test_pdf = str(pdf_files[0])
    st.sidebar.info(f"대상 파일: {os.path.basename(test_pdf)}")

    # 2. 버튼으로 테스트 시나리오 트리거
    if st.sidebar.button("🎨 하이라이트 주입 테스트"):
        # 가상의 검색 결과 생성 (첫 페이지 텍스트 일부)
        import fitz
        with fitz.open(test_pdf) as doc:
            page_text = doc[0].get_text()
            # 첫 50자를 검색어로 설정
            sample_text = page_text.strip()[:50]
        
        from langchain_core.documents import Document
        mock_docs = [Document(page_content=sample_text, metadata={"page": 1})]
        
        # 좌표 추출 실행
        annotations = get_pdf_annotations(test_pdf, mock_docs)
        
        if annotations:
            # 세션 상태에 주입 (실제 UI 로직과 동일)
            st.session_state.pdf_annotations = annotations
            # 자동 이동 트리거를 위해 페이지 설정
            st.session_state.pdf_page_index = annotations[0]["page"]
            st.success(f"✅ {len(annotations)}개의 하이라이트 데이터를 주입했습니다. (페이지 {st.session_state.pdf_page_index})")
        else:
            st.error("❌ 좌표 추출에 실패했습니다.")

    if st.sidebar.button("🧹 초기화"):
        st.session_state.pdf_annotations = []
        st.session_state.pdf_page_index = 1
        st.rerun()

    # 3. 뷰어 영역 (src/ui/ui.py의 로직을 간소화하여 재현)
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("데이터 디버거")
        st.write("현재 세션 상태 (st.session_state):")
        st.json({
            "pdf_page_index": st.session_state.get("pdf_page_index", 1),
            "pdf_annotations_count": len(st.session_state.get("pdf_annotations", []))
        })
        if st.session_state.get("pdf_annotations"):
            st.write("상세 좌표:")
            st.write(st.session_state.pdf_annotations)

    with c2:
        st.subheader("PDF 렌더링 확인")
        pdf_bytes = open(test_pdf, "rb").read()
        
        # 실제 UI 로직과 동일한 파라미터로 호출
        viewer_params = {
            "input": pdf_bytes,
            "height": 800,
            "pages_to_render": [st.session_state.get("pdf_page_index", 1)],
        }
        
        annotations = st.session_state.get("pdf_annotations", [])
        if annotations:
            viewer_params["annotations"] = annotations
            
        pdf_viewer(**viewer_params)

if __name__ == "__main__":
    test_ui_render()
