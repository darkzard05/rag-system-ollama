"""
PDF 문서 로딩 및 텍스트 추출을 담당하는 모듈.
PyMuPDF4LLM을 사용하여 초고속으로 구조적 마크다운을 추출하며 RAG 최적화 청킹을 수행합니다.
"""

import hashlib
import logging
from collections.abc import Callable

from langchain_core.documents import Document

from common.exceptions import (
    EmptyPDFError,
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
    import pymupdf4llm

    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log(
                "📑 문서 구조 분석 및 마크다운 변환 중",
                session_id=session_id,
            )
            if on_progress:
                on_progress()

            # PyMuPDF4LLM 최적화 호출
            try:
                from common.config import PARSING_CONFIG

                # 설정값 추출 (기본값 포함)
                write_images = PARSING_CONFIG.get("write_images", False)
                fontsize_limit = PARSING_CONFIG.get("fontsize_limit", 3)
                ignore_code = PARSING_CONFIG.get("ignore_code", False)
                extract_words = PARSING_CONFIG.get("extract_words", True)
                ignore_graphics = PARSING_CONFIG.get("ignore_graphics", True)
                table_strategy = PARSING_CONFIG.get("table_strategy", "fast")

                # 마크다운 변환 실행
                chunks = pymupdf4llm.to_markdown(
                    file_path,
                    page_chunks=True,
                    write_images=write_images,
                    fontsize_limit=fontsize_limit,
                    ignore_code=ignore_code,
                    extract_words=extract_words,
                    ignore_graphics=ignore_graphics,
                    table_strategy=table_strategy,
                )

                docs = []
                for i, chunk in enumerate(chunks):
                    metadata = chunk.get("metadata", {})
                    page_num = metadata.get("page", i + 1)

                    # [수정] 단어 좌표(words) 정보를 추출하여 메타데이터에 보관
                    # 형식: (x0, y0, x1, y1, "text") 리스트
                    raw_words = chunk.get("words", [])
                    formatted_words = [
                        (w[0], w[1], w[2], w[3], w[4]) for w in raw_words
                    ]

                    doc = Document(
                        page_content=chunk.get("text", ""),
                        metadata={
                            "source": file_name,
                            "page": page_num,
                            "total_pages": metadata.get("page_count", len(chunks)),
                            "engine": "pymupdf4llm",
                            "format": "markdown",
                            "chunk_index": i,
                            "is_already_chunked": True,
                            "word_coords": formatted_words,  # 좌표 데이터 직접 저장
                            "has_coordinates": len(formatted_words) > 0,
                        },
                    )
                    docs.append(doc)

                total_chars = sum(len(doc.page_content) for doc in docs)
                SessionManager.add_status_log(
                    f"문서 분석 완료 ({len(docs)} 페이지, 약 {total_chars:,}자)",
                    session_id=session_id,
                )

            except Exception as e:
                logger.error(f"PyMuPDF4LLM 처리 실패: {e}")
                raise

            if not docs:
                raise EmptyPDFError(
                    filename=file_name, details={"reason": "추출된 텍스트가 없습니다."}
                )

            # 성능 지표 기록 (토큰 수 대략 계산)
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs

        except Exception as e:
            logger.error(f"[RAG] [PDF] 최종 오류: {e}")
            if isinstance(e, EmptyPDFError):
                raise
            raise PDFProcessingError(
                message=str(e), details={"filename": file_name}
            ) from e
