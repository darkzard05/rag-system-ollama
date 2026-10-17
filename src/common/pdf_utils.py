"""PDF 파일 열기/페이지 수 조회 공용 유틸리티. (R: Group 7 중복 통합)

``document_processor``, ``viewer``, ``document_hydrator``, ``common.utils`` 에
흩어져 있던 ``fitz.open`` 컨텍스트 매니저를 단일 진원으로 통합한다.
각 하위 호출부의 **예외 처리 / 캐시 계약은 그대로 보존**하기 위해,
열기(``open_pdf_document``)와 페이지 수 조회(``get_pdf_page_count``)를
별도 순수 헬퍼로 제공한다.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)


def _fitz() -> Any:
    """PyMuPDF 모듈을 지연 임포트해 반환한다 (성능 최적화)."""
    import pymupdf

    return pymupdf


@contextlib.contextmanager
def open_pdf_document(file_path: str) -> Iterator[Any]:
    """PDF 파일을 자동으로 닫아주는 컨텍스트 매니저.

    ``document_processor`` 의 기존 구현을 승격한 공용 버전 (R: Group 7).
    재시도/예외 경로에서도 안전하게 리소스를 정리한다.
    ``with open_pdf_document(path) as doc:`` 형태로 사용한다.
    """
    fitz = _fitz()
    doc = None
    try:
        doc = fitz.open(file_path)
        yield doc
    finally:
        if doc:
            try:
                doc.close()
            except Exception as e:
                logger.warning(f"[PDF] 파일 종료 중 오류 ({file_path}): {e}")


def _count_pages(
    opener: Callable[[], Any], source: str, *, exc_info: bool = False
) -> int | None:
    """문서를 열어 페이지 수를 세고, 실패는 로깅 후 ``None``으로 흡수한다.

    호출부가 로깅 강도(``exc_info``)와 ``source`` 라벨을 지정한다.
    절대 raise하지 않는다.
    """
    try:
        with opener() as doc:
            return len(doc)
    except Exception as e:
        logger.error(f"PDF 페이지 수 조회 실패 ({source}): {e}", exc_info=exc_info)
        return None


def get_pdf_page_count(file_path: str) -> int | None:
    """PDF 총 페이지 수를 반환한다. 열 수 없거나 손상된 경우 ``None``.

    - 파일이 없으면 ``None``
    - ``viewer._get_pdf_total_pages`` 의 "절대 raise하지 않음" 계약을 보존한다.
    - **캐시 계층은 포함하지 않는다** (호출자가 st.cache_data 등으로 감싼다).
    """
    if not file_path:
        return None
    fitz = _fitz()
    return _count_pages(lambda: fitz.open(file_path), file_path, exc_info=True)


def get_pdf_page_count_from_bytes(content: bytes) -> int | None:
    """메모리 내 PDF 바이트의 페이지 수를 반환한다. 파싱 불가 시 ``None``.

    Tier1.5: 업로드 시점 압축폭탄(고페이지·고압축) 조기 거부용. 렌더링 없이
    xref 테이블만 읽으므로 가볍다. 절대 raise하지 않는다.
    """
    if not content:
        return None
    fitz = _fitz()
    return _count_pages(
        lambda: fitz.open(stream=content, filetype="pdf"),
        f"bytes {len(content)}B",
    )
