"""
README.md의 동적 섹션(기술 스택, 프로젝트 구조 등)을 자동으로 업데이트하는 스크립트입니다.
"""

import os
import re
from pathlib import Path

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
README_PATH = ROOT_DIR / "readme.md"

def get_tech_stack():
    """requirements/base.txt에서 주요 버전 정보를 추출합니다."""
    base_req_path = ROOT_DIR / "requirements" / "base.txt"
    if not base_req_path.exists():
        return "- Data not available (requirements/base.txt not found)"
    
    try:
        with open(base_req_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return "- Data not available (Error reading file)"
    
    # 추출할 주요 패키지 목록
    packages = {
        "Streamlit": r"streamlit==([\d\.]+)",
        "LangChain": r"langchain>=([\d\.]+)",
        "LangGraph": r"langgraph>=([\d\.]+)",
        "PyMuPDF4LLM": r"pymupdf4llm==([\d\.]+)",
        "Ollama": r"ollama==([\d\.]+)",
        "FastAPI": r"fastapi==([\d\.]+)",
    }
    
    lines = []
    for name, pattern in packages.items():
        match = re.search(pattern, content)
        version = match.group(1) if match else "latest"
        lines.append(f"- **{name}**: {version}")
    
    return "\\n".join(lines)

def get_project_tree():
    """주요 소스 폴더의 구조를 텍스트로 생성합니다."""
    folders = ["src", "scripts", "tests"]
    tree_lines = ["rag-system-ollama/"]
    
    for folder in folders:
        folder_path = ROOT_DIR / folder
        if not folder_path.exists():
            continue
            
        tree_lines.append(f"├── {folder}/")
        
        # 1단계 하위 폴더/파일만 표시 (가독성 위해)
        try:
            items = sorted([item for item in folder_path.iterdir() if not item.name.startswith("__")])
            for i, item in enumerate(items):
                is_last = (i == len(items) - 1)
                prefix = "│   └── " if is_last else "│   ├── "
                
                suffix = "/" if item.is_dir() else ""
                comment = ""
                
                # 간단한 주석 추가 (중요 파일 위주)
                if item.name == "main.py": comment = " # 🏁 Entry Point"
                elif item.name == "rag_core.py": comment = " # 🧠 RAG Engine"
                
                tree_lines.append(f"{prefix}{item.name}{suffix}{comment}")
        except Exception:
            tree_lines.append(f"│   └── (Error reading {folder})")
            
    return "```text\\n" + "\\n".join(tree_lines) + "\\n```"

def update_readme():
    """Magic Tags를 찾아 내용을 교체합니다."""
    if not README_PATH.exists():
        print("README.md not found.")
        return

    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Tech Stack 업데이트
        stack_data = get_tech_stack().replace("\\n", "\n")
        content = re.sub(
            r"(<!-- TECH_STACK_START -->).*?(<!-- TECH_STACK_END -->)",
            f"\\1\\n{stack_data}\\n\\2",
            content,
            flags=re.DOTALL
        )

        # 2. Project Tree 업데이트
        tree_data = get_project_tree().replace("\\n", "\n")
        content = re.sub(
            r"(<!-- TREE_START -->).*?(<!-- TREE_END -->)",
            f"\\1\\n{tree_data}\\n\\2",
            content,
            flags=re.DOTALL
        )

        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ README.md has been automatically updated with latest project info.")
    except Exception as e:
        print(f"❌ Error updating README: {e}")

if __name__ == "__main__":
    update_readme()
