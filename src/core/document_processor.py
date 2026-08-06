"""
PDF 문서 로딩 및 텍스트 추출을 담당하는 모듈.
PyMuPDF4LLM을 사용하여 초고속으로 구조적 마크다운을 추출하며 RAG 최적화 청킹을 수행합니다.
"""

import asyncio
import contextlib
import hashlib
import logging
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


def _get_monitor() -> Any:
    """성능 모니터를 첫 사용 시점에 지연 초기화합니다."""
    return get_performance_monitor()


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


@contextlib.contextmanager
def open_pdf_document(file_path: str):
    """PDF 파일을 자동으로 정리하는 컨텍스트 매니저.

    모든 재시도 경로에서 안전하게 리소스를 정리합니다.
    """
    import fitz

    doc = None
    try:
        doc = fitz.open(file_path)
        yield doc
    finally:
        if doc:
            try:
                doc.close()
                logger.debug(f"[RAG] [PDF] 파일 핸들 정리 완료: {file_path}")
            except Exception as e:
                logger.warning(f"[RAG] [PDF] 파일 종료 중 오류: {e}")


def _extraction_progress_pct(page_number: int, total_pages: int) -> int:
    """텍스트 추출 단계의 중간 진행률 (5% → 45%)을 계산합니다."""
    if total_pages <= 0:
        return 45
    return round(5 + 40 * (page_number / total_pages))


def _extract_markdown_via_thread(
    file_path: str,
    *,
    extract_words: bool,
    margins: list,
    table_strategy: str,
    graphics_limit: int,
    fontsize_limit: int,
    ignore_code: bool,
    write_images: bool,
) -> list:
    """PyMuPDF4LLM 마크다운 추출을 워커 스레드에서 실행하기 위한 sync 래퍼.

    fitz Document는 스레드 간 공유가 안전하지 않으므로 파일 경로로 자체 열어 사용합니다.
    """
    import pymupdf4llm

    with open_pdf_document(file_path) as doc:
        try:
            return pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                extract_words=extract_words,
                table_strategy=table_strategy,
                graphics_limit=graphics_limit,
                fontsize_limit=fontsize_limit,
                ignore_code=ignore_code,
                write_images=write_images,
                margins=margins,
            )
        except Exception as layout_error:
            logger.warning(
                f"[RAG] [PDF] PyMuPDF4LLM 추출 오류 ({layout_error}). 테이블 제외 후 재시도합니다."
            )
            return pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                extract_words=extract_words,
                table_strategy="lines",
                graphics_limit=0,
                fontsize_limit=fontsize_limit,
                ignore_code=ignore_code,
                write_images=False,
                margins=margins,
            )


async def load_pdf_docs(
    file_path: str,
    file_name: str,
    on_progress: Callable[[int], Any] | None = None,
    file_bytes: bytes | None = None,
    session_id: str | None = None,
    file_hash: str | None = None,
) -> list[Document]:
    """
    PyMuPDF4LLM을 사용하여 문서를 페이지 단위 마크다운으로 변환하고 RAG용 Document 객체 리스트를 생성합니다.
    """
    from cache.coord_cache import coord_cache
    from common.config import HYDRATION_MODE, PARSING_CONFIG

    logger.info(f"[RAG] [LOAD] PDF 분석 시작: {file_name} (Mode: {HYDRATION_MODE})")
    with _get_monitor().track_operation(
        OperationType.PDF_LOADING, {"file": file_name}
    ) as op:
        try:
            SessionManager.add_status_log(
                "문서 구조 분석 및 마크다운 변환 중",
                session_id=session_id,
            )

            # context manager로 안전하게 PDF 핸들 관리
            with open_pdf_document(file_path) as doc:
                total_pages = len(doc)
                if file_hash is None:
                    file_hash = compute_file_hash(file_path, file_bytes)

                # 설정값 로드
                target_margins = PARSING_CONFIG.get("margins", [0, 72, 0, 72])
                table_strategy = PARSING_CONFIG.get("table_strategy", "lines_strict")

                # 하이드레이션 모드에 관계없이 정밀 하이라이트를 위해 단어 추출 활성화
                do_extract_words = HYDRATION_MODE != "none"

                try:
                    chunks = await asyncio.to_thread(
                        _extract_markdown_via_thread,
                        file_path,
                        extract_words=do_extract_words,
                        margins=target_margins,
                        table_strategy=table_strategy,
                        graphics_limit=PARSING_CONFIG.get("graphics_limit", 5000),
                        fontsize_limit=PARSING_CONFIG.get("fontsize_limit", 3),
                        ignore_code=PARSING_CONFIG.get("ignore_code", False),
                        write_images=PARSING_CONFIG.get("write_images", False),
                    )
                except Exception as second_error:
                    # 3차 시도: ONNXRuntimeError 등 예측 모델 라이브러리 충돌 시 폴백 안전 장치 가동
                    logger.warning(
                        f"[RAG] [PDF] PyMuPDF4LLM 최종 실패: {second_error}. "
                        "호환성이 우수한 표준 PyMuPDF(C-Engine) 텍스트 추출 모드로 안전하게 자동 전환합니다."
                    )
                    SessionManager.add_status_log(
                        "시스템 호환성 엔진(Classic C-Engine)으로 자동 전환 중...",
                        session_id=session_id,
                    )

                    chunks = []
                    for page_idx in range(total_pages):
                        page = doc[page_idx]
                        page_num = page_idx + 1

                        # C-Engine을 통한 안정적인 텍스트 추출
                        text = page.get_text("text")

                        # 정밀 하이라이팅을 위한 단어 좌표 목록 추출
                        words = []
                        if do_extract_words:
                            try:
                                raw_words = page.get_text("words")
                                words = [
                                    (w[0], w[1], w[2], w[3], w[4]) for w in raw_words
                                ]
                            except Exception as word_e:
                                logger.debug(f"단어 좌표 추출 실패 (스킵): {word_e}")

                        chunks.append(
                            {
                                "text": text,
                                "metadata": {"page": page_num},
                                "words": words,
                                "tables": [],
                            }
                        )
                        if on_progress:
                            on_progress(_extraction_progress_pct(page_num, total_pages))

                if on_progress:
                    on_progress(_extraction_progress_pct(total_pages, total_pages))

                docs: list[Document] = []
                current_section = "Introduction/Root"

                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        text = chunk.get("text", "")
                        metadata = chunk.get("metadata", {})
                        page_num = metadata.get("page", i + 1)
                        toc_items = chunk.get("toc_items", [])
                    else:
                        text = str(chunk)
                        metadata = {}
                        page_num = i + 1
                        toc_items = []

                    if toc_items:
                        current_section = toc_items[-1][1]

                    # 하이드레이션 모드에 따른 메타데이터 및 좌표 캐싱
                    has_coords = HYDRATION_MODE != "none"
                    bbox = None

                    # 단어 좌표가 추출된 경우 즉시 캐시 저장
                    if isinstance(chunk, dict) and "words" in chunk and chunk["words"]:
                        await coord_cache.save_coords(
                            file_hash, page_num, chunk["words"]
                        )

                    if HYDRATION_MODE == "precision_clip":
                        # 페이지 전체 Rect를 기본 bbox로 설정 (폴백용)
                        page_rect = doc[page_num - 1].rect
                        bbox = [page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1]

                    tables = chunk.get("tables", []) if isinstance(chunk, dict) else []

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

            # context manager 종료로 doc.close() 자동 실행

            SessionManager.add_status_log(
                f"문서 분석 완료: 총 {len(docs)}페이지 지식 확보",
                session_id=session_id,
            )
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs

        except Exception as e:
            logger.error(f"[RAG] [PDF] 추출 오류: {e}", exc_info=True)
            # 정리 작업 수행
            try:
                if op:
                    pass
            except Exception as cleanup_e:
                logger.error(f"[RAG] [PDF] 정리 작업 중 오류 발생: {cleanup_e}")
            raise PDFProcessingError(
                message=str(e), details={"filename": file_name}
            ) from e
