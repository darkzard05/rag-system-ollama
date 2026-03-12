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

    logger.info(f"[RAG] [LOAD] PDF 분석 시작: {file_name}")
    with monitor.track_operation(OperationType.PDF_LOADING, {"file": file_name}) as op:
        try:
            SessionManager.add_status_log(
                "문서 구조 분석 및 마크다운 변환 중 (Layout Mode)",
                session_id=session_id,
            )

            # 1. [정밀 분석] TOC(목차) 및 폰트 기반 헤더 식별
            doc = fitz.open(file_path)
            total_pages = len(doc)
            logger.info(f"[RAG] [LOAD] PDF 열기 성공 (총 {total_pages} 페이지)")
            ref_start_page = 999999

            # [고도화] 폰트 분석 기반 지능형 헤더 감지 (Layout Mode 보조)
            from common.config import PARSING_CONFIG

            # 학술 논문 특성에 맞춰 본문 폰트 크기 임계값을 더 엄격하게 설정 (보통 9-11pt가 본문)
            body_limit = PARSING_CONFIG.get("body_limit", 12)
            max_levels = PARSING_CONFIG.get(
                "max_levels", 3
            )  # H1~H3까지만 인식하여 노이즈 억제

            try:
                # [고도화] 폰트 기반 헤더 정보 생성
                if hasattr(pymupdf4llm, "IdentifyHeaders"):
                    hdr_info = pymupdf4llm.IdentifyHeaders(
                        doc, body_limit=body_limit, max_levels=max_levels
                    )
                elif hasattr(pymupdf4llm, "TocHeaders"):
                    # 구버전 폴백
                    hdr_info = pymupdf4llm.TocHeaders(doc)
                else:
                    hdr_info = None

                # TOC 정보와 결합 (TOC가 있으면 우선 활용)
                toc = doc.get_toc()
                # 참고문헌 시작 페이지 파악
                for entry in toc:
                    title = str(entry[1]).lower()
                    if any(
                        kw in title for kw in ["references", "bibliography", "참고문헌"]
                    ):
                        ref_start_page = entry[2]
                        break

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
                        f"문서 구조 분석: {ref_start_page}페이지부터 참고문헌 섹션 식별",
                        session_id=session_id,
                    )
            except Exception as e:
                logger.debug(f"헤더/섹션 분석 실패: {e}")
                hdr_info = None

            if on_progress:
                on_progress()

            # 2. PyMuPDF4LLM 고도화 호출 (캐시 최적화 포함)
            from cache.coord_cache import coord_cache

            file_hash = compute_file_hash(file_path, file_bytes)

            target_margins = PARSING_CONFIG.get("margins", (0, 60, 0, 60))

            # [최적화] 페이지 특성 샘플링 기반 전략 최적화
            # (모든 페이지를 루프 돌면 느려지므로 주요 페이지 샘플링 진단)
            sample_pages = [0, total_pages // 2, total_pages - 1]
            layout_results = [
                _detect_page_layout(doc[i]) for i in sample_pages if i < total_pages
            ]

            # 다수가 다단(Multi-column)이면 폰트 제한 및 레이아웃 분석 강화
            is_global_multi = (
                sum(1 for r in layout_results if r["is_multi_column"]) >= 1
            )
            best_table_strategy = "lines"
            if any(r["strategy"] == "text" for r in layout_results):
                best_table_strategy = "text"  # 선 없는 표 우선 대응

            logger.info(
                f"[RAG] [LOAD] 레이아웃 진단: MultiColumn={is_global_multi}, TableStrategy={best_table_strategy}"
            )

            chunks = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                write_images=PARSING_CONFIG.get("write_images", False),
                fontsize_limit=PARSING_CONFIG.get("fontsize_limit", 3),
                ignore_code=PARSING_CONFIG.get("ignore_code", False),
                extract_words=PARSING_CONFIG.get("extract_words", True),
                ignore_graphics=False,
                force_text=True,  # [수정] 에러 방지를 위해 True로 설정
                table_strategy=best_table_strategy,  # [동적 적용]
                hdr_info=hdr_info,
                header=False,
                footer=False,
                margins=target_margins,
            )
            logger.info(
                f"[RAG] [LOAD] 마크다운 변환 완료 ({len(chunks)}개 페이지 조각 생성)"
            )

            doc.close()

            docs: list[Document] = []
            # ... (중략: 목차 및 참고문헌 필터링 로직)
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

                # A. TOC(목차) 페이지 필터링
                if page_num <= max(3, total_pages // 10) and any(
                    p in lower_text[:100] for p in TOC_PATTERNS
                ):
                    continue

                # B. 참고문헌 섹션 관리
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

                # C. [최적화 핵심] 좌표 데이터 캐시 오프로딩
                raw_words = chunk.get("words", [])
                has_coords = False
                if raw_words:
                    # 실제 리스트는 외부 캐시에 보관
                    formatted_words = [
                        (w[0], w[1], w[2], w[3], w[4]) for w in raw_words
                    ]
                    coord_cache.save_coords(file_hash, page_num, formatted_words)
                    has_coords = True

                # 인덱스용 Document에는 최소한의 정보만 보관
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,  # UI용 파일 경로 복구
                            "file_hash": file_hash,  # 좌표 복구용 키
                            "page": page_num,
                            "total_pages": total_pages,
                            "engine": "pymupdf4llm-layout-optimized",
                            "format": "markdown",
                            "chunk_index": len(docs),
                            "has_coordinates": has_coords,
                            # word_coords 리스트는 명시적으로 제외 (메모리 절감)
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
