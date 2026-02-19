import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.document_processor import load_pdf_docs

def evaluate_extraction():
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    file_name = os.path.basename(test_pdf)
    print(f"[*] Evaluating: {file_name}")
    try:
        docs = load_pdf_docs(test_pdf, file_name)
        output_path = ROOT_DIR / "logs" / "extraction_debug_v1.txt"
        os.makedirs(ROOT_DIR / "logs", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs[:5]):
                header = f"--- PAGE {i+1} ---\n"
                f.write(header)
                f.write(doc.page_content)
                f.write("\n\n")
        print(f"[+] Saved to: {output_path}")
        print("\n[Sample View - Page 1]")
        print("-" * 50)
        print(docs[0].page_content[:800])
        print("-" * 50)
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    evaluate_extraction()
