import os
import time
import fitz
import pymupdf4llm
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

def old_extraction_logic(file_path):
    doc = fitz.open(file_path)
    text_content = []
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        lines = []
        for b in blocks:
            if b[6] == 0:
                lines.append(b[4].strip())
        text_content.append("\n\n".join(lines))
    doc.close()
    return "\n\n".join(text_content)

def new_extraction_logic(file_path):
    return pymupdf4llm.to_markdown(file_path)

def run_comparison():
    print(f"[*] Starting Comparison: {os.path.basename(TEST_PDF)}")
    start_v1 = time.time()
    v1_text = old_extraction_logic(TEST_PDF)
    time_v1 = time.time() - start_v1
    start_v2 = time.time()
    v2_text = new_extraction_logic(TEST_PDF)
    time_v2 = time.time() - start_v2
    print("-" * 55)
    print(f"{'Metric':<20} | {'V1 (Old)':<15} | {'V2 (New)':<15}")
    print("-" * 55)
    print(f"{'Time (Seconds)':<20} | {time_v1:<15.4f} | {time_v2:<15.4f}")
    print(f"{'Char Count':<20} | {len(v1_text):<15} | {len(v2_text):<15}")
    print("-" * 55)
    idx_v1 = v1_text.find("ABSTRACT")
    print("\n[V1 Sample]")
    print(v1_text[idx_v1:idx_v1+300] if idx_v1 != -1 else "N/A")
    idx_v2 = v2_text.find("ABSTRACT")
    print("\n[V2 Sample]")
    print(v2_text[idx_v2:idx_v2+300] if idx_v2 != -1 else "N/A")

if __name__ == "__main__":
    run_comparison()
