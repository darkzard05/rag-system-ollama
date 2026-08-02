import os
import sys
import cProfile
import pstats
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

# Document 클래스 위치 찾기
# grep -r "class Document" src/
# 결과: src/core/document.py
from core.document import Document
from core.document_hydrator import hydrate_documents

# 테스트용 문서 생성
docs = [
    Document(metadata={"file_path": str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf"), "has_coordinates": True, "file_hash": "test_hash", "page": i})
    for i in range(1, 6)
]

# 프로파일링
profiler = cProfile.Profile()
profiler.enable()
hydrate_documents(docs)
profiler.disable()

# 결과 저장
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
