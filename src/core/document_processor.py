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
    [고도화] Layout Mode를 활성화하여 AI 기반 헤더/푸터 제거 및 구조 분석을 수행합니다.
    """
    # [중요] 레이아웃 모드 활성화를 위해 pymupdf.layout을 pymupdf4llm보다 먼저 임포트
    try:
        import pymupdf.layout  # noqa: F401
    except ImportError:
        logger.warning(
            "pymupdf.layout 모듈을 로드할 수 없습니다. 일반 모드로 동작합니다."
        )

    import fitz  # PyMuPDF
    import pymupdf4llm

    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log(
                "문서 구조 분석 및 마크다운 변환 중 (Layout Mode)",
                session_id=session_id,
            )

            # 1. [정밀 분석] TOC(목차) 기반 섹션 식별
            doc = fitz.open(file_path)
            ref_start_page = 999999
            toc_info = None

            try:
                # [안전] 속성 존재 여부 확인 후 호출
                toc = doc.get_toc()
                if toc and hasattr(pymupdf4llm, "TocHeaders"):
                    toc_info = pymupdf4llm.TocHeaders(doc)

                    # 참고문헌 시작 페이지 파악
                    for entry in toc:
                        title = str(entry[1]).lower()
                        if any(
                            kw in title
                            for kw in ["references", "bibliography", "참고문헌"]
                        ):
                            ref_start_page = entry[2]
                            break
                else:
                    # [폴백] TOC가 없는 경우 폰트 분석 기반 헤더 감지 수행
                    SessionManager.add_status_log(
                        "문서 목차가 없어 폰트 기반 지능형 헤더 분석을 수행합니다.",
                        session_id=session_id,
                    )
                    if hasattr(pymupdf4llm, "IdentifyHeaders"):
                        toc_info = pymupdf4llm.IdentifyHeaders(doc)

                if ref_start_page == 999999:
                    # 목차에 없으면 마지막 5페이지에서 패턴 검색
                    for p_idx in range(len(doc) - 1, max(0, len(doc) - 5), -1):
                        page_text = doc[p_idx].get_text().lower()
                        if any(
                            kw in page_text[:500]
                            for kw in ["references", "bibliography", "참고문헌"]
                        ):
                            ref_start_page = p_idx + 1
                            break

                if ref_start_page != 999999:
                    SessionManager.add_status_log(
                        f"문서 구조 분석: {ref_start_page}페이지부터 참고문헌 섹션을 식별했습니다.",
                        session_id=session_id,
                    )
            except Exception as e:
                logger.debug(f"섹션 분석 실패: {e}")

            if on_progress:
                on_progress()

            # 2. PyMuPDF4LLM 최적화 호출 (Layout Mode)
            from common.config import PARSING_CONFIG

            # [핵심] 파일 경로를 직접 전달하고 header=False, footer=False 설정
            chunks = pymupdf4llm.to_markdown(
                file_path,
                page_chunks=True,
                write_images=PARSING_CONFIG.get("write_images", False),
                fontsize_limit=PARSING_CONFIG.get("fontsize_limit", 3),
                ignore_code=PARSING_CONFIG.get("ignore_code", False),
                # [최적화] 인덱싱 시에는 단어 좌표를 추출하지 않음 (용량 및 메모리 절약)
                # 하이라이트가 필요한 시점에만 utils.py의 On-demand 로직으로 추출
                extract_words=False,
                ignore_graphics=PARSING_CONFIG.get("ignore_graphics", True),
                table_strategy="fast",  # 레이아웃 모드 최적화 전략
                hdr_info=toc_info,  # 식별된 헤더 정보 적용
                header=False,  # AI 기반 헤더 자동 제거
                footer=False,  # AI 기반 푸터 자동 제거
                margins=0,  # 수동 마진 무시 (AI 판단 우선)
            )

            # 열린 doc 객체 정리
            doc.close()

            docs: list[Document] = []
            reference_started = False
            reference_text_buffer = []

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
                    continue

                # B. 참고문헌(References) 식별 및 추출
                if page_num >= ref_start_page:
                    if not reference_started:
                        reference_started = True
                    reference_text_buffer.append(text)

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
                    reference_started = True
                    reference_text_buffer.append(text)

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
                            "file_path": file_path,
                            "page": page_num,
                            "total_pages": total_pages,
                            "engine": "pymupdf4llm-layout",
                            "format": "markdown",
                            "chunk_index": len(docs),
                            "word_coords": formatted_words,
                            "has_coordinates": len(formatted_words) > 0,
                        },
                    )
                )

            # D. [핵심] 레퍼런스 맵 생성 및 본문 링크 연결
            from common.utils import parse_reference_section

            all_ref_text = "\n".join(reference_text_buffer)
            ref_map = parse_reference_section(all_ref_text)

            if ref_map:
                import re

                num_pattern = re.compile(r"[\[\(](\d+)[\]\)]")
                name_year_pattern = re.compile(
                    r"([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*[\(,]?\s*(\d{4})[\s\)]?"
                )

                for doc in docs:
                    linked_refs = {}
                    found_nums = num_pattern.findall(doc.page_content)
                    for num in found_nums:
                        if num in ref_map:
                            linked_refs[num] = ref_map[num]

                    found_authors = name_year_pattern.findall(doc.page_content)
                    for author, year in found_authors:
                        clean_author = author.replace(" et al.", "").strip().lower()
                        key = f"{clean_author}_{year}"
                        if key in ref_map:
                            linked_refs[f"{author} ({year})"] = ref_map[key]

                    if linked_refs:
                        doc.metadata["linked_references"] = linked_refs

                SessionManager.add_status_log(
                    f"인용 분석 완료: 총 {len(ref_map)}개의 참고문헌 식별",
                    session_id=session_id,
                )

            filtered_count = len(chunks) - len(docs)
            SessionManager.add_status_log(
                f"문서 분석 완료: 총 {len(docs)}페이지 지식 확보 (정제됨: {filtered_count}p)",
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
