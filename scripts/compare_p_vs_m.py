import os
from markitdown import MarkItDown
import pymupdf4llm
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

def compare():
    print("--- PyMuPDF4LLM ---")
    md_p = pymupdf4llm.to_markdown(TEST_PDF, write_images=False, graphics_limit=100)
    idx_p = md_p.find("ABSTRACT")
    if idx_p != -1:
        print(md_p[idx_p:idx_p+200])
    
    print("\n--- MarkItDown ---")
    mid = MarkItDown()
    res = mid.convert(TEST_PDF)
    md_m = res.text_content
    idx_m = md_m.find("ABSTRACT")
    if idx_m != -1:
        print(md_m[idx_m:idx_m+200])

if __name__ == "__main__":
    compare()