"""
PDF 문서 로딩 및 텍스트 추출을 담당하는 모듈.
최신 Docling 라이브러리를 사용하여 고품질 구조적 마크다운을 추출하며, 실패 시 PyMuPDF4LLM으로 폴백합니다.
"""

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

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


def _load_with_pymupdf(file_path: str) -> list[Document]:
    """PyMuPDF4LLM을 사용한 폴백 로딩"""
    import pymupdf4llm

    logger.info("[Fallback] PyMuPDF4LLM을 사용하여 문서 변환 시도")
    md_text = pymupdf4llm.to_markdown(file_path)

    return [
        Document(
            page_content=md_text,
            metadata={
                "source": Path(file_path).name,
                "format": "markdown",
                "engine": "pymupdf4llm",
            },
        )
    ]


def load_pdf_docs(
    file_path: str,
    file_name: str,
    on_progress: Callable[[], None] | None = None,
    file_bytes: bytes | None = None,
    session_id: str | None = None,
) -> list[Document]:
    """
    DoclingLoader를 사용하여 고품질 구조적 마크다운을 추출합니다.
    """
    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log(
                "AI 기반 정밀 구조 분석 중 (Docling)", session_id=session_id
            )
            if on_progress:
                on_progress()

            # Docling 시도
            try:
                from langchain_docling import DoclingLoader

                from core.model_loader import ModelManager

                # [최적화] 전역 관리되는 DocumentConverter 인스턴스를 사용하여 설정(OCR 등)을 일관되게 적용
                converter = ModelManager.get_docling_converter()

                # [최적화] DoclingLoader는 내부적으로 Docling의 강력한 분석 엔진을 사용하여
                # 표와 레이아웃을 마크다운으로 완벽하게 보존합니다.
                loader = DoclingLoader(
                    file_path=file_path,
                    converter=converter,
                    export_type="markdown",
                )

                docs = loader.load()

                # 메타데이터 보강
                for doc in docs:
                    doc.metadata.update(
                        {"source": file_name, "engine": "docling", "format": "markdown"}
                    )

                SessionManager.replace_last_status_log(
                    f"Docling 분석 완료 ({len(docs)} 청크)",
                    session_id=session_id,
                )

            except ImportError:
                logger.warning(
                    "Docling 라이브러리가 없습니다. PyMuPDF4LLM으로 전환합니다."
                )
                docs = _load_with_pymupdf(file_path)
                SessionManager.replace_last_status_log(
                    "PyMuPDF 분석 완료 (Docling 미설치)", session_id=session_id
                )
            except Exception as e:
                logger.error(f"Docling 처리 실패: {e}. PyMuPDF4LLM으로 전환합니다.")
                docs = _load_with_pymupdf(file_path)
                SessionManager.replace_last_status_log(
                    "PyMuPDF 분석 완료 (Docling 실패)", session_id=session_id
                )

            if not docs:
                raise EmptyPDFError(
                    filename=file_name, details={"reason": "유효한 텍스트가 없습니다."}
                )

            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs

        except Exception as e:
            logger.error(f"[RAG] [PDF] 최종 오류: {e}")
            raise PDFProcessingError(
                message=str(e), details={"filename": file_name}
            ) from e
