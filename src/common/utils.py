"""utils.py — 프로젝트 공용 유틸리티 집계 모듈 (Batch 5.4 재구성).

도메인별 구현은 하위 모듈로 분리되었으며, 모든 기존 import 경로를 유지하기
위해 본 모듈에서 재수출합니다:
- ``utils.py._utils_text``: LaTeX/인용/전처리/토큰 카운트
- ``utils.py._utils_annotations``: PDF 주석/툴팁 렌더링
- ``utils.py._utils_hash``: 고속/안정/파일 해시
- ``utils.py._utils_cache``: Streamlit 캐시 래퍼
- ``utils.py._utils_async``: 백그라운드 워커
"""

from common._utils_annotations import (
    _build_line_boxes,
    apply_tooltips_to_response,
    extract_annotations_from_docs,
)
from common._utils_async import run_in_background_worker
from common._utils_cache import safe_cache_data, safe_cache_resource
from common._utils_hash import compute_file_hash, doc_stable_id, fast_hash
from common._utils_text import (
    count_tokens_rough,
    normalize_latex_delimiters,
    preprocess_text,
    strip_context_tokens,
)

__all__ = [
    "apply_tooltips_to_response",
    "compute_file_hash",
    "count_tokens_rough",
    "doc_stable_id",
    "extract_annotations_from_docs",
    "fast_hash",
    "normalize_latex_delimiters",
    "preprocess_text",
    "run_in_background_worker",
    "safe_cache_data",
    "safe_cache_resource",
    "strip_context_tokens",
    "_build_line_boxes",
]
