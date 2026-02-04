
import fitz

def create_test_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    
    # 짧은 문장과 긴 문장이 섞인 텍스트
    text = """
    제1장 개요.
    이것은 테스트 문서입니다.
    짧음.
    아주 짧은 문장.
    네.
    아니오.
    이 문장은 병합 테스트를 위해 충분히 길게 작성된 문장입니다.
    그러므로 병합되지 않아야 합니다.
    하지만.
    이것은 병합되어야 합니다.
    끝.
    """
    
    page.insert_text((50, 50), text, fontsize=12)
    doc.save(path)
    doc.close()
    print(f"Created: {path}")

if __name__ == "__main__":
    import os
    os.makedirs("tests/data", exist_ok=True)
    create_test_pdf("tests/data/test_chunking.pdf")
