from reportlab.pdfgen import canvas
from pathlib import Path
import os

def create_dummy_pdf():
    # 저장 경로 설정
    base_dir = Path(__file__).parent.parent.parent / "data" / "temp"
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / "test_highlight.pdf"
    
    # PDF 생성
    c = canvas.Canvas(str(file_path))
    c.drawString(100, 750, "This is a test document for highlighting verification.")
    c.drawString(100, 730, "The second line contains important information about the project.")
    c.drawString(100, 710, "Coordinate extraction is critical for user experience.")
    c.save()
    
    print(f"Created dummy PDF at: {file_path}")
    return str(file_path)

if __name__ == "__main__":
    create_dummy_pdf()