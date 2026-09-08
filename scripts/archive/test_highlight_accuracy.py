
import os
import sys
import logging
import fitz  # PyMuPDF
import re
import pymupdf4llm
from langchain_core.documents import Document

# 프로젝트 루트를 path에 추가하여 src 모듈을 임포트 가능하게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from common.utils import extract_annotations_from_docs

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def clean_markdown_for_search(text):
    """하이라이트 검색을 위해 마크다운 기호 제거"""
    # **bold**, _italic_, [link](url) 등 제거
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 특수 기호들 정제
    text = text.replace('\\_', '_').replace('\\*', '*')
    return text

def run_test_scenario(name, doc, pdf_path, page_idx=0):
    print(f"\n{'='*20} {name} {'='*20}")
    
    # 마크다운 정제 시도
    original_content = doc.page_content
    doc.page_content = clean_markdown_for_search(original_content)
    
    print(f"정제된 텍스트 요약: {doc.page_content[:100]}...")
    
    annotations = extract_annotations_from_docs([doc])
    
    if annotations:
        print(f"✅ 성공: {len(annotations)}개 영역 추출됨")
        
        # 시각화 저장
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports'))
        os.makedirs(output_dir, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        output_path = os.path.join(output_dir, f"highlight_{safe_name}.png")
        
        with fitz.open(pdf_path) as pdf:
            page = pdf[page_idx]
            for anno in annotations:
                if anno['page'] == page_idx + 1:
                    rect = fitz.Rect(anno['x'], anno['y'], anno['x'] + anno['width'], anno['y'] + anno['height'])
                    page.add_highlight_annot(rect)
            
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            pix.save(output_path)
            print(f"📸 결과 이미지: {output_path}")
        return True
    else:
        print("❌ 실패: 하이라이트 영역이 생성되지 않음")
        return False

def main():
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tests/data/2201.07520v1.pdf'))
    if not os.path.exists(pdf_path):
        print(f"파일 없음: {pdf_path}")
        return

    # 시나리오 1: On-demand (마크다운 정제 적용)
    content_with_md = "CM3: A **CAUSAL MASKED MULTIMODAL MODEL** OF THE INTERNET"
    doc_ondemand = Document(
        page_content=content_with_md,
        metadata={"file_path": pdf_path, "page": 1}
    )
    run_test_scenario("Scenario_1_OnDemand_Cleaned", doc_ondemand, pdf_path)

    # 시나리오 2: Metadata-based (실제 좌표 추출 활성화)
    print("\n[INFO] PyMuPDF4LLM으로 좌표를 포함하여 로딩 중...")
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, extract_words=True)
    
    if chunks:
        chunk = chunks[0]
        text = chunk.get("text", "")
        raw_words = chunk.get("words", [])
        # Document 포맷에 맞게 변환
        formatted_words = [(w[0], w[1], w[2], w[3], w[4]) for w in raw_words]
        
        doc_metadata = Document(
            page_content=text[100:500], # 중간 부분 텍스트 슬라이싱
            metadata={
                "file_path": pdf_path, 
                "page": 1,
                "word_coords": formatted_words
            }
        )
        run_test_scenario("Scenario_2_Metadata_Sequence_Success", doc_metadata, pdf_path)
    else:
        print("청크 로드 실패")

if __name__ == "__main__":
    main()
