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
                "문서 구조 분석 및 마크다운 변환 중",
                session_id=session_id,
            )

            # 1. [정밀 분석] TOC(목차) 기반 헤더 감지 및 마진 설정
            doc = fitz.open(file_path)
            ref_start_page = 999999
            toc_info = None

            try:
                # A. TOC 정보를 활용한 헤더 감지기 초기화 (PyMuPDF4LLM 최신 기능)
                toc = doc.get_toc()
                if toc:
                    toc_info = pymupdf4llm.TocHeaders(doc)

                    # B. 참고문헌 시작 페이지 파악
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
                    toc_info = pymupdf4llm.IdentifyHeaders(doc)

                if ref_start_page == 999999:
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
                logger.debug(f"TOC/섹션 분석 실패: {e}")

            if on_progress:
                on_progress()

            # 2. PyMuPDF4LLM 최적화 호출
            from common.config import PARSING_CONFIG

            # [최적화] margins 설정 (72 points = 1 inch). 상하 1인치씩 무시하여 헤더/푸터 제거 시도
            # 사용자가 설정에서 변경할 수 있도록 config와 연동 가능
            margins = PARSING_CONFIG.get(
                "margins", (0, 72, 0, 72)
            )  # (left, top, right, bottom)

            chunks = pymupdf4llm.to_markdown(
                doc,  # 파일 경로 대신 이미 열린 doc 객체 전달 (리소스 절약)
                page_chunks=True,
                write_images=PARSING_CONFIG.get("write_images", False),
                fontsize_limit=PARSING_CONFIG.get("fontsize_limit", 3),
                ignore_code=PARSING_CONFIG.get("ignore_code", False),
                extract_words=PARSING_CONFIG.get(
                    "extract_words", True
                ),  # 하이라이트 기능을 위해 True 권장
                ignore_graphics=PARSING_CONFIG.get("ignore_graphics", True),
                table_strategy=PARSING_CONFIG.get("table_strategy", "lines_strict"),
                hdr_info=toc_info,  # TOC 기반 헤더 감지 적용
                margins=margins,  # 헤더/푸터 제외를 위한 마진 적용
            )

            # doc 객체 닫기는 pymupdf4llm 내부에서 처리되거나 컨텍스트 종료 시 처리됨 (여기서는 명시적으로 닫지 않음)

            docs: list[Document] = []
            reference_started = False
            reference_text_buffer = []  # 레퍼런스 섹션 텍스트 보관용

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
                        f"불필요한 목차 페이지({page_num}p)를 제외합니다.",
                        session_id=session_id,
                    )
                    continue

                # B. 참고문헌(References) 식별 및 추출
                if page_num >= ref_start_page:
                    if not reference_started:
                        SessionManager.add_status_log(
                            f"지식 정제: {page_num}페이지 이후의 참고문헌 섹션을 식별했습니다.",
                            session_id=session_id,
                        )
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
                    SessionManager.add_status_log(
                        f"지식 정제: 패턴 기반으로 참고문헌 섹션을 감지했습니다 ({page_num}p~)",
                        session_id=session_id,
                    )
                    reference_started = True
                    reference_text_buffer.append(text)

                if reference_started:
                    # 참고문헌 시작된 이후의 페이지는 텍스트만 버퍼에 넣고 본문(docs)에는 추가하지 않음
                    continue

                # C. 데이터 보관 (좌표 정보 포함)
                raw_words = chunk.get("words", [])
                formatted_words = [(w[0], w[1], w[2], w[3], w[4]) for w in raw_words]

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,  # 절대 경로 추가
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

            # D. [핵심] 레퍼런스 맵 생성 및 본문 링크 연결
            from common.utils import parse_reference_section

            all_ref_text = "\n".join(reference_text_buffer)
            ref_map = parse_reference_section(all_ref_text)

            if ref_map:
                import re

                # 1. 숫자형 [1], (1) 감지
                num_pattern = re.compile(r"[\[\(](\d+)[\]\)]")
                # 2. 이름-연도형 Author et al. (2024) 또는 Author (2024) 또는 (Author, 2024) 감지
                name_year_pattern = re.compile(
                    r"([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*[\(,]?\s*(\d{4})[\s\)]?"
                )

                for doc in docs:
                    linked_refs = {}

                    # 숫자형 인용 매칭
                    found_nums = num_pattern.findall(doc.page_content)
                    for num in found_nums:
                        if num in ref_map:
                            linked_refs[num] = ref_map[num]

                    # 이름-연도형 인용 매칭
                    found_authors = name_year_pattern.findall(doc.page_content)
                    for author, year in found_authors:
                        clean_author = author.replace(" et al.", "").strip().lower()
                        key = f"{clean_author}_{year}"
                        if key in ref_map:
                            linked_refs[f"{author} ({year})"] = ref_map[key]

                    # 메타데이터에 주입
                    if linked_refs:
                        doc.metadata["linked_references"] = linked_refs

                SessionManager.add_status_log(
                    f"인용 분석 완료: 총 {len(ref_map)}개의 참고문헌을 식별하여 본문에 연결했습니다.",
                    session_id=session_id,
                )

            filtered_count = len(chunks) - len(docs)
            SessionManager.add_status_log(
                f"문서 분석 완료: 총 {len(docs)}페이지의 유효 지식을 확보했습니다. ({filtered_count}페이지 정제됨)",
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
