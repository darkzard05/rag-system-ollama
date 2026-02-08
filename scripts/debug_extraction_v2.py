import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()

def evaluate_new_extraction():
    import pymupdf4llm
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    print(f"[*] Evaluating New Extraction (Markdown): {os.path.basename(test_pdf)}")
    try:
        md_text = pymupdf4llm.to_markdown(test_pdf)
        output_path = ROOT_DIR / "logs" / "extraction_debug_v2.md"
        os.makedirs(ROOT_DIR / "logs", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[+] Saved to: {output_path}")
        print("\n[Sample View]")
        print("-" * 50)
        print(md_text[:800])
        print("-" * 50)
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    evaluate_new_extraction()