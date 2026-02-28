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
    [고도화] TOC 분석을 통해 참고문헌 섹션을 정밀 필터링합니다.
    """
    import fitz  # PyMuPDF
    import pymupdf4llm

    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log(
                "📑 문서 구조 분석 및 마크다운 변환 중",
                session_id=session_id,
            )

            # 1. [정밀 분석] TOC(목차) 기반 참고문헌 시작 페이지 파악
            ref_start_page = 999999
            try:
                with fitz.open(file_path) as pdf:
                    toc = pdf.get_toc()
                    for entry in toc:
                        title = str(entry[1]).lower()
                        if any(
                            kw in title
                            for kw in ["references", "bibliography", "참고문헌"]
                        ):
                            ref_start_page = entry[2]
                            SessionManager.add_status_log(
                                f"📂 문서 구조 분석: {ref_start_page}페이지부터 참고문헌 섹션을 식별했습니다.",
                                session_id=session_id,
                            )
                            break
            except Exception as e:
                logger.debug(f"TOC 분석 실패: {e}")

            if on_progress:
                on_progress()

            # 2. PyMuPDF4LLM 최적화 호출
            from common.config import PARSING_CONFIG

            chunks = pymupdf4llm.to_markdown(
                file_path,
                page_chunks=True,
                write_images=PARSING_CONFIG.get("write_images", False),
                fontsize_limit=PARSING_CONFIG.get("fontsize_limit", 3),
                ignore_code=PARSING_CONFIG.get("ignore_code", False),
                extract_words=PARSING_CONFIG.get("extract_words", True),
                ignore_graphics=PARSING_CONFIG.get("ignore_graphics", True),
                table_strategy=PARSING_CONFIG.get("table_strategy", "fast"),
            )

            docs: list[Document] = []
            reference_started = False

            TOC_PATTERNS = ["table of contents", "contents", "목차"]
            REF_PATTERNS = ["references", "bibliography", "참고문헌"]

            for i, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                lower_text = text.lower().strip()
                metadata = chunk.get("metadata", {})
                page_num = metadata.get("page", i + 1)
                total_pages = metadata.get("page_count", len(chunks))

                # A. TOC(목차) 페이지 필터링 (앞부분 10% 이내)
                if page_num <= max(3, total_pages // 10) and any(
                    p in lower_text[:100] for p in TOC_PATTERNS
                ):
                    SessionManager.add_status_log(
                        f"🧹 불필요한 목차 페이지({page_num}p)를 제외합니다.",
                        session_id=session_id,
                    )
                    continue

                # B. 참고문헌(References) 필터링
                if page_num >= ref_start_page:
                    if not reference_started:
                        SessionManager.add_status_log(
                            f"🚫 지식 정제: {page_num}페이지 이후의 참고문헌 섹션을 제외합니다.",
                            session_id=session_id,
                        )
                        reference_started = True

                elif (
                    not reference_started
                    and page_num > (total_pages * 0.7)
                    and any(
                        f"## {p}" in lower_text
                        or f"**{p}**" in lower_text
                        or lower_text.startswith(p)
                        for p in REF_PATTERNS
                    )
                ):
                    SessionManager.add_status_log(
                        f"🚫 지식 정제: 텍스트 패턴 기반으로 참고문헌 섹션을 감지하여 제외합니다 ({page_num}p~)",
                        session_id=session_id,
                    )
                    reference_started = True

                if reference_started:
                    continue

                # C. 데이터 보관 (좌표 정보 포함)
                raw_words = chunk.get("words", [])
                formatted_words = [(w[0], w[1], w[2], w[3], w[4]) for w in raw_words]

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_num,
                            "total_pages": total_pages,
                            "engine": "pymupdf4llm",
                            "format": "markdown",
                            "chunk_index": len(docs),
                            "word_coords": formatted_words,
                            "has_coordinates": len(formatted_words) > 0,
                        },
                    )
                )

            filtered_count = len(chunks) - len(docs)
            SessionManager.add_status_log(
                f"✅ 문서 분석 완료: 총 {len(docs)}페이지의 유효 지식을 확보했습니다. ({filtered_count}페이지 정제됨)",
                session_id=session_id,
            )
            op.tokens = sum(len(doc.page_content.split()) for doc in docs)
            return docs

        except Exception as e:
            logger.error(f"[RAG] [PDF] 최종 오류: {e}")
            if isinstance(e, EmptyPDFError):
                raise
            raise PDFProcessingError(
                message=str(e), details={"filename": file_name}
            ) from e
