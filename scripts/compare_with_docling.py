import os
import time
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

import pymupdf4llm
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

def test_pymupdf(file_path):
    print(f"\n--- Testing PyMuPDF4LLM (Strategy: text) ---")
    start = time.time()
    try:
        # table_strategy="text"를 사용하여 테두리 없는 표 인식 시도
        md_text = pymupdf4llm.to_markdown(
            file_path, 
            table_strategy="text",
            graphics_limit=1000
        )
        elapsed = time.time() - start
        print(f"Elapsed Time: {elapsed:.2f}s")
        print(f"Text Length: {len(md_text)} chars")
        return md_text, elapsed
    except Exception as e:
        print(f"Error: {e}")
        return "", 0

def test_docling_no_ocr(file_path):
    print(f"\n--- Testing IBM Docling (OCR OFF) ---")
    start = time.time()
    try:
        # [최적화] OCR 비활성화 및 레이아웃 분석만 수행
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True # 표 인식은 유지
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        elapsed = time.time() - start
        print(f"Elapsed Time: {elapsed:.2f}s")
        print(f"Text Length: {len(text)} chars")
        return text, elapsed
    except Exception as e:
        print(f"Error: {e}")
        return "", 0

def main():
    test_file = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    if not os.path.exists(test_file):
        print(f"Error: File not found at {test_file}")
        return

    print(f"Comparing extraction (OCR OFF) for: {os.path.basename(test_file)}")
    
    p_text, p_time = test_pymupdf(test_file)
    d_text, d_time = test_docling_no_ocr(test_file)

    # 샘플 결과 저장 (비교용)
    output_dir = ROOT_DIR / "logs" / "extraction_comparison_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir / "pymupdf_v2.md", "w", encoding="utf-8") as f:
        f.write(p_text) # 좀 더 길게 저장
    
    with open(output_dir / "docling_no_ocr.md", "w", encoding="utf-8") as f:
        f.write(d_text)

    print(f"\nSamples saved to {output_dir}")
    print("\nSummary (OCR OFF):")
    if p_time > 0 and d_time > 0:
        print(f"- Speed Gap: PyMuPDF is {d_time/p_time:.1f}x faster" if d_time > p_time else f"- Speed: Docling is {p_time/d_time:.1f}x faster")
    print(f"- Volume: Docling extracted {len(d_text) - len(p_text)} more characters" if len(d_text) > len(p_text) else f"- Volume: PyMuPDF extracted {len(p_text) - len(d_text)} more characters")

if __name__ == "__main__":
    main()
