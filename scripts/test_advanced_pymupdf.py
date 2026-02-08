import os
import time
import pymupdf4llm
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
TEST_PDF = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")

def test():
    print("Testing PyMuPDF4LLM Advanced Params...")
    s = time.time()
    chunks = pymupdf4llm.to_markdown(
        TEST_PDF,
        page_chunks=True,
        write_images=False,
        ignore_graphics=True,
        table_strategy="lines",
        fontsize_limit=3
    )
    dur = time.time() - s
    
    print("--- Results ---")
    line1 = "Time: " + str(round(dur, 2)) + "s"
    print(line1)
    
    line2 = "Chunks: " + str(len(chunks))
    print(line2)
    
    if chunks:
        c = chunks[0]
        txt_len = len(c.get('text', ''))
        tbl_cnt = len(c.get('tables', []))
        print("Page 1 Text Length: " + str(txt_len))
        print("Page 1 Tables: " + str(tbl_cnt))

if __name__ == "__main__":
    test()