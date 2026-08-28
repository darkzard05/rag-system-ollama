"""
PDF 문서 로딩 및 텍스트 추출을 담당하는 모듈.
PyMuPDF4LLM을 사용하여 초고속으로 구조적 마크다운을 추출하며 RAG 최적화 청킹을 수행합니다.
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


def _to_float(value: Any) -> float | None:
    """값을 float로 변환합니다. 파싱 불가(None 포함) 시 None을 반환합니다."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _words_bbox(words: list[Any]) -> list[float] | None:
    """단어 좌표 리스트의 실제 텍스트 영역 bbox ``[x0, y0, x1, y1]``를 계산합니다.

    tuple/리스트(5-tuple·8-tuple)와 dict 형식을 모두 지원하며, 파싱 불가 항목은
    건너뜁니다. 유효 좌표가 하나도 없으면 None을 반환해 호출부가 페이지 rect로
    폴백하도록 합니다.
    """
    xs0: list[float] = []
    ys0: list[float] = []
    xs1: list[float] = []
    ys1: list[float] = []
    for w in words:
        if isinstance(w, dict):
            x0 = _to_float(w.get("x0"))
            y0 = _to_float(w.get("y0"))
            x1 = _to_float(w.get("x1"))
            y1 = _to_float(w.get("y1"))
        elif isinstance(w, (tuple, list)) and len(w) >= 4:
            x0 = _to_float(w[0])
            y0 = _to_float(w[1])
            x1 = _to_float(w[2])
            y1 = _to_float(w[3])
        else:
            continue
        if x0 is None or y0 is None or x1 is None or y1 is None:
            continue
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    if not xs0:
        return None
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


@contextlib.contextmanager
def open_pdf_document(file_path: str):
    """PDF 파일을 자동으로 정리하는 컨텍스트 매니저.

    모든 재시도 경로에서 안전하게 리소스를 정리합니다.
    """
    import pymupdf as fitz

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


# 단글자 볼드 드롭캡 정규화: "**M** odeling" -> "Modeling".
# PyMuPDF4LLM이 헤딩의 볼드 단글자를 "**X** " 형태로 추출하여 단어 중간에
# 공백이 끼워지는 왜곡을 유발하므로, 청킹/임베딩 전에 단글자 볼드만 좁게 교정한다.
# 멀티글자 볼드 (**CM3**)나 강조는 보존한다.
_RE_BOLD_INITIAL = re.compile(r"\*\*([A-Za-z])\*\*\s+")


def _normalize_markdown_bold_dropcaps(markdown: str) -> str:
    """단글자 볼드 드롭캡 패턴을 교정합니다 (예: '**M** odeling' -> 'Modeling')."""
    if not markdown:
        return markdown
    return _RE_BOLD_INITIAL.sub(r"\1", markdown)


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

    def _extract() -> list:
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

    with open_pdf_document(file_path) as doc:
        try:
            chunks: Any = _extract()
        except Exception as layout_error:
            logger.warning(
                f"[RAG] [PDF] PyMuPDF4LLM 추출 오류 ({layout_error}). 테이블 제외 후 재시도합니다."
            )
            chunks = pymupdf4llm.to_markdown(
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
    # 청킹/임베딩 전 단글자 볼드 드롭캡 왜곡 교정 (F2)
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                chunk["text"] = _normalize_markdown_bold_dropcaps(chunk["text"])
    return chunks


def _extract_page_c_engine(
    file_path: str,
    page_idx: int,
    do_extract_words: bool,
) -> dict[str, Any]:
    """C-Engine 단일 페이지 추출 (스레드 안전: 스레드별 PDF 핸들 사용).

    PyMuPDF ``Document``/``page`` 객체는 스레드 간 공유가 안전하지 않으므로,
    워커는 반드시 자체 ``open_pdf_document`` 핸들을 열고 닫는다. 메인 스레드의
    ``doc`` 핸들을 절대 공유하지 않는다.
    """
    page_num = page_idx + 1
    with open_pdf_document(file_path) as page_doc:
        page = page_doc[page_idx]
        # C-Engine을 통한 안정적인 텍스트 추출
        text = page.get_text("text")

        # 정밀 하이라이팅을 위한 단어 좌표 목록 추출
        words: list[tuple[float, float, float, float, str]] = []
        if do_extract_words:
            try:
                raw_words = page.get_text("words")
                words = [
                    (float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]))
                    for w in raw_words
                ]
            except Exception as word_e:
                logger.debug(f"단어 좌표 추출 실패 (스킵): {word_e}")

    return {
        "text": text,
        "metadata": {"page": page_num},
        "words": words,
        "tables": [],
    }


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
    with get_performance_monitor().track_operation(
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

                    # 스레드 풀 기반 병렬 페이지 추출.
                    # PyMuPDF Document는 스레드 간 공유가 안전하지 않으므로 각 워커가
                    # 자체 PDF 핸들(``open_pdf_document``)을 열고 닫는다. 메인 스레드의
                    # ``doc``는 절대 공유하지 않는다. 페이지 순서는 인덱스 순으로 보장.
                    max_workers = min(total_pages, os.cpu_count() or 4) or 1
                    # markdown 성공 경로(line 238)와 동일 변수 chunks 를 공유하여
                    # 하단 enumerate(chunks) 에서 양 경로 결과를统一 소비한다.
                    # 타입 어노테이션을 생략해 mypy no-redef(재정의 불일치)를 회피한다.
                    chunks = [None] * total_pages
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_idx = {
                            executor.submit(
                                _extract_page_c_engine,
                                file_path,
                                page_idx,
                                do_extract_words,
                            ): page_idx
                            for page_idx in range(total_pages)
                        }
                        # as_completed로 완료 순서와 무관하게 페이지 순서로 재조립
                        for future in future_to_idx:
                            page_idx = future_to_idx[future]
                            try:
                                chunks[page_idx] = future.result()
                            except Exception as page_e:
                                logger.error(
                                    f"[RAG] [PDF] C-Engine 페이지 {page_idx + 1} 추출 실패: "
                                    f"{page_e}"
                                )
                                chunks[page_idx] = {
                                    "text": "",
                                    "metadata": {"page": page_idx + 1},
                                    "words": [],
                                    "tables": [],
                                }
                            # 진행률 콜백은 메인 스레드에서만 호출(스레드 안전)
                            if on_progress:
                                on_progress(
                                    _extraction_progress_pct(page_idx + 1, total_pages)
                                )

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
                        # [R2-06] 페이지 전체 rect 대신 실제 텍스트 영역(단어 bbox 합집합)으로
                        # bbox를 산출 — 헤더/푸터(마진) 영역을 제외해 캐시 미스 시
                        # 불필요한 전체 페이지 I/O를 줄입니다. 실질 하이라이트 정밀도는
                        # utils.py의 토큰 시퀀스 매칭이 담당합니다 (config 주석 참조).
                        raw_words = (
                            chunk.get("words", []) if isinstance(chunk, dict) else []
                        )
                        content_bbox = _words_bbox(raw_words)
                        if content_bbox is not None:
                            bbox = content_bbox
                        else:
                            page_rect = doc[page_num - 1].rect
                            bbox = [
                                page_rect.x0,
                                page_rect.y0,
                                page_rect.x1,
                                page_rect.y1,
                            ]

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
                                # [R2-04] 페이지 단위 추출 결과는 사전 분할 단위 —
                                # split_documents가 과대 페이지만 하위 분할하고
                                # 나머지를 재사용하도록 플래그를 세팅한다.
                                "is_already_chunked": True,
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
