"""
PDF 문서 로딩 및 텍스트 추출을 담당하는 모듈.
PyMuPDF4LLM을 사용하여 초고속으로 구조적 마크다운을 추출하며 RAG 최적화 청킹을 수행합니다.
"""

import hashlib
import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from common.exceptions import (
    PDFProcessingError,
)
from core.session import SessionManager
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


def compute_file_hash(file_path: str, data: bytes | None = None) -> str:
    """파일 또는 데이터의 SHA256 해시를 계산합니다."""
    sha256_hash = hashlib.sha256()
    try:
        if data is not None:
            sha256_hash.update(data)
        else:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"해시 계산 실패: {e}")
        return ""


def _detect_page_layout(page) -> dict[str, Any]:
    """
    페이지의 텍스트 및 선(Line) 분포를 분석하여 최적의 파싱 전략을 제안합니다.
    """
    text_blocks = page.get_text("blocks")
    paths = page.get_drawings()

    # 1. 테이블 선 밀도 체크
    horizontal_lines = [
        p
        for p in paths
        if p["type"] == "l" and abs(p["items"][0][0][1] - p["items"][0][1][1]) < 2
    ]

    # 2. 다단(Multi-column) 여부 체크 (x좌표 분포 분석)
    x_coords = [b[0] for b in text_blocks if len(b) > 4 and isinstance(b[4], str)]
    is_multi_column = False
    if len(x_coords) > 10:
        mid_x = page.rect.width / 2
        left_count = sum(1 for x in x_coords if x < mid_x * 0.8)
        right_count = sum(1 for x in x_coords if x > mid_x * 1.2)
        if left_count > 5 and right_count > 5:
            is_multi_column = True

    # 전략 결정
    strategy = "lines"  # 기본값
    if len(horizontal_lines) < 3 and any(
        re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s+){2,}", b[4])
        for b in text_blocks
        if len(b) > 4
    ):
        # 선은 없는데 숫자가 나열된 경우 (Borderless Table)
        strategy = "text"

    return {
        "strategy": strategy,
        "is_multi_column": is_multi_column,
        "has_tables": len(horizontal_lines) > 5 or strategy == "text",
    }


def load_pdf_docs(
    file_path: str,
    file_name: str,
    on_progress: Callable[[], None] | None = None,
    file_bytes: bytes | None = None,
    session_id: str | None = None,
) -> list[Document]:
    """
    PyMuPDF4LLM을 사용하여 문서를 페이지 단위 마크다운으로 변환하고 RAG용 Document 객체 리스트를 생성합니다.
    """
    import fitz  # PyMuPDF
    import pymupdf4llm

    from cache.coord_cache import coord_cache
    from common.config import HYDRATION_MODE, PARSING_CONFIG

    logger.info(f"[RAG] [LOAD] PDF 분석 시작: {file_name} (Mode: {HYDRATION_MODE})")
    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log(
                "문서 구조 분석 및 마크다운 변환 중",
                session_id=session_id,
            )

            doc = fitz.open(file_path)
            total_pages = len(doc)
            file_hash = compute_file_hash(file_path, file_bytes)

            # 설정값 로드
            target_margins = PARSING_CONFIG.get("margins", [0, 72, 0, 72])
            table_strategy = PARSING_CONFIG.get("table_strategy", "lines_strict")

            # [최적화] 하이드레이션 모드에 따른 추출 전략 결정
            do_extract_words = HYDRATION_MODE == "full"

            chunks = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                extract_words=do_extract_words,
                table_strategy=table_strategy,
                graphics_limit=PARSING_CONFIG.get("graphics_limit", 5000),
                fontsize_limit=PARSING_CONFIG.get("fontsize_limit", 3),
                ignore_code=PARSING_CONFIG.get("ignore_code", False),
                write_images=PARSING_CONFIG.get("write_images", False),
                margins=target_margins,
            )

            if on_progress:
                on_progress()

            docs: list[Document] = []
            current_section = "Introduction/Root"

            for i, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                page_num = metadata.get("page", i + 1)

                toc_items = chunk.get("toc_items", [])
                if toc_items:
                    current_section = toc_items[-1][1]

                # [수정] 하이드레이션 모드에 따른 메타데이터 주입
                has_coords = HYDRATION_MODE != "none"
                bbox = None

                if HYDRATION_MODE == "precision_clip":
                    page_rect = doc[page_num - 1].rect
                    bbox = [page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1]
                elif HYDRATION_MODE == "full" and "words" in chunk:
                    # 이미 전체 추출된 경우 즉시 캐시 저장
                    coord_cache.save_coords(file_hash, page_num, chunk["words"])

                tables = chunk.get("tables", [])

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,
                            "file_hash": file_hash,
                            "page": page_num,
                            "total_pages": total_pages,
                            "engine": f"pymupdf4llm-{HYDRATION_MODE}",
                            "current_section": current_section,
                            "has_coordinates": has_coords,
                            "bbox": bbox,
                            "has_tables": len(tables) > 0,
                            "table_count": len(tables),
                            "chunk_index": len(docs),
                        },
                    )
                )

            doc.close()

            SessionManager.add_status_log(
                f"문서 분석 완료: 총 {len(docs)}페이지 지식 확보",
                session_id=session_id,
            )
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs

        except Exception as e:
            logger.error(f"[RAG] [PDF] 추출 오류: {e}")
            raise PDFProcessingError(
                message=str(e), details={"filename": file_name}
            ) from e
