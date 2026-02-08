"""
PDF 문서 로딩 및 텍스트 추출을 담당하는 모듈.
최적화: pymupdf-layout 패키지와 graphics_limit=100 설정을 통해 추출 속도를 극대화합니다.
"""

import hashlib
import logging
from collections.abc import Callable

from langchain_core.documents import Document

from common.exceptions import (
    EmptyPDFError,
    PDFProcessingError,
)
from common.utils import preprocess_text
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
    PyMuPDF4LLM을 사용하여 고품질 마크다운을 최적화된 속도로 추출합니다.
    최적화: ignore_graphics=True 설정을 통해 속도를 10배 이상 향상시키고, table_strategy를 통해 품질을 보강합니다.
    """
    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log(
                "초고속 구조적 분석 중", session_id=session_id
            )
            if on_progress:
                on_progress()

            try:
                import gc

                import pymupdf4llm

                # [고도화 적용] 최신 pymupdf4llm 옵션 반영
                # table_strategy: 'lines_strict'는 선이 명확한 표를 더 정확하게 인식합니다.
                # graphics_limit: 너무 많은 벡터 패스가 있는 페이지는 속도 저하를 방지하기 위해 처리를 제한합니다.
                pages_data = pymupdf4llm.to_markdown(
                    file_path,
                    page_chunks=True,
                    write_images=False,
                    ignore_graphics=True,
                    graphics_limit=500,  # 500개 이상의 벡터 패스가 있는 경우 그래픽 무시
                    table_strategy="lines_strict",
                    fontsize_limit=3,
                    show_progress=False,
                )

                if not pages_data:
                    raise EmptyPDFError(
                        filename=file_name,
                        details={"reason": "추출 데이터가 없습니다."},
                    )

                total_pages = len(pages_data)
                docs = []

                for i, chunk in enumerate(pages_data):
                    text = chunk.get("text", "")
                    if text:
                        clean_text = preprocess_text(text)
                        if clean_text and len(clean_text) > 10:
                            meta = chunk.get("metadata", {})
                            docs.append(
                                Document(
                                    page_content=clean_text,
                                    metadata={
                                        "source": file_name,
                                        "page": meta.get("page", i + 1),
                                        "total_pages": total_pages,
                                        "format": "markdown",
                                    },
                                )
                            )

                    if (i + 1) % 10 == 0 or (i + 1) == total_pages:
                        SessionManager.replace_last_status_log(
                            f"구조 분석 중 ({i + 1}/{total_pages}p)",
                            session_id=session_id,
                        )

                # 추출 완료 후 즉시 메모리 정리 유도
                del pages_data
                gc.collect()

            except Exception as e:
                logger.error(f"추출 실패: {e}")
                raise PDFProcessingError(
                    filename=file_name, details={"reason": str(e)}
                ) from e

            if not docs:
                raise EmptyPDFError(
                    filename=file_name, details={"reason": "유효한 텍스트가 없습니다."}
                )

            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs

        except Exception as e:
            logger.error(f"[RAG] [LOAD] 최종 오류: {e}")
            raise PDFProcessingError(
                filename=file_name, details={"reason": str(e)}
            ) from e
