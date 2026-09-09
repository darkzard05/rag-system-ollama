"""
문서 좌표 하이드레이션 모듈.
검색 결과 문서의 word_coords를 캐시에서 복구하거나 즉시 추출(Lazy)합니다.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langchain_core.documents import Document

from cache.coord_cache import coord_cache
from common.exceptions import CoordCacheReadError

logger = logging.getLogger(__name__)


def _extract_page_words_sync(
    path: str, page_num: int, chunk_bbox: Any | None
) -> list[Any] | None:
    """PDF 파일에서 특정 페이지의 단어 좌표를 동기로 추출합니다 (스레드 푸시용)."""
    import pymupdf as fitz  # lazy: 좌표 추출 시에만 import

    try:
        with fitz.open(path) as doc_obj:
            page_obj = doc_obj[page_num - 1]
            if chunk_bbox:
                raw_words = page_obj.get_text("words", clip=fitz.Rect(chunk_bbox))
            else:
                raw_words = page_obj.get_text("words")
            return [(w[0], w[1], w[2], w[3], w[4]) for w in raw_words]
    except IndexError:
        logger.warning(f"[HYDRATE] 페이지 인덱스 초과: P{page_num}")
        return None
    except (OSError, fitz.FileDataError) as e:
        logger.error(f"[HYDRATE] 파일 처리 중 오류 ({os.path.basename(path)}): {e}")
        return None


async def _hydrate_page(
    doc: Document,
    coords_map: dict[int, list[dict[str, Any]]],
    page_num: int,
) -> tuple[int, list[Any] | None, bool]:
    """단일 페이지 좌표를 캐시에서 읽거나 병렬로 추출합니다.

    반환: (page_num, coords, was_fresh). was_fresh=True는 이 분기가 실제
    추출(캐시 미스)을 수행했음을, 캐시 적중이면 False를 의미한다. bbox
    폴백 포함 양쪽 시도 모두 fresh로 간주하며, 추출 실패 시 예외 없이
    (page_num, None, True)를 반환해 해당 페이지가 그냥 좌표 없이 남게
    한다. 저장(``coord_cache.save_coords``)은 호출하지 않는다.
    """
    path = doc.metadata["file_path"]
    coords = coords_map.get(page_num)
    if coords:
        return page_num, coords, False

    logger.info(f"[HYDRATE] 정밀 좌표 추출: {os.path.basename(path)} P{page_num}")
    chunk_bbox = doc.metadata.get("bbox")
    coords = await asyncio.to_thread(
        _extract_page_words_sync, path, page_num, chunk_bbox
    )
    # Fallback: bbox-scoped extract failed -> retry whole page
    if not coords and chunk_bbox is not None:
        coords = await asyncio.to_thread(_extract_page_words_sync, path, page_num, None)
    return page_num, coords, True


async def hydrate_documents(docs: list[Document]) -> None:
    """문서 리스트의 좌표 데이터를 캐시에서 복구하거나, 없으면 즉시 추출(Lazy)합니다.

    각 파일을 1회만 열어 해당 파일의 모든 대상 문서(청크)를 일괄 처리하여
    I/O 비용을 최소화합니다.
    """
    # 1. 파일별로 처리 대상 문서 그룹화
    file_path_map: dict[str, list[Document]] = {}
    for doc in docs:
        if "word_coords" in doc.metadata or not doc.metadata.get("has_coordinates"):
            continue

        path = doc.metadata.get("file_path")
        if path and os.path.exists(path):
            if path not in file_path_map:
                file_path_map[path] = []
            file_path_map[path].append(doc)

    if not file_path_map:
        return

    # 2. 대상 문서(청크)별 좌표 복원 및 비동기 스레드 파싱
    for path, target_docs in file_path_map.items():
        # 1. 파일별로 처리 대상 문서 그룹화 (file_hash 기준)
        file_hash_map: dict[str, list[Document]] = {}
        for doc in target_docs:
            file_hash = doc.metadata.get("file_hash")
            if file_hash:
                if file_hash not in file_hash_map:
                    file_hash_map[file_hash] = []
                file_hash_map[file_hash].append(doc)

        # 2. 대상 문서(청크)별 좌표 복원 및 비동기 스레드 파싱
        for file_hash, docs_in_file in file_hash_map.items():
            page_nums: list[int] = []
            seen_pages: set[int] = set()
            for doc in docs_in_file:
                pages = doc.metadata.get("pages") or (
                    [doc.metadata["page"]]
                    if doc.metadata.get("page") is not None
                    else []
                )
                for page in pages:
                    if page not in seen_pages:
                        seen_pages.add(page)
                        page_nums.append(page)
            try:
                coords_map = await coord_cache.get_coords_batch(file_hash, page_nums)
            except CoordCacheReadError as e:
                logger.warning(
                    f"[HYDRATE] 좌표 캐시 읽기 실패 ({os.path.basename(path)}, "
                    f"file_hash={file_hash}): {e}"
                )
                # 실패를 명시적 per-file 마커로 기록하고 다른 파일 수화는 계속한다.
                for doc in docs_in_file:
                    doc.metadata["coord_cache_error"] = True
                continue

            for doc in docs_in_file:
                pages = doc.metadata.get("pages") or (
                    [doc.metadata["page"]]
                    if doc.metadata.get("page") is not None
                    else []
                )
                page_coords: dict[int, list] = {}

                # 파일의 페이지들을 병렬로 캐시-읽기/추출한다. 저장(fresh)
                # 여부는 gather 결과를 페이지 순서대로 재조립할 때 결정한다.
                results = await asyncio.gather(
                    *(_hydrate_page(doc, coords_map, page_num) for page_num in pages)
                )
                result_map: dict[int, tuple[list[Any] | None, bool]] = {
                    page_num: (coords, was_fresh)
                    for page_num, coords, was_fresh in results
                }

                for page_num in pages:
                    coords, was_fresh = result_map[page_num]
                    # 캐시 적중 페이지는 재저장하지 않는다 (기존 추출 데이터
                    # 보존 + 불필요한 I/O 방지). 추출(fresh) 성공 시에만 저장.
                    if was_fresh and coords is not None:
                        await coord_cache.save_coords(file_hash, page_num, coords)
                    if coords:
                        page_coords[page_num] = coords

                doc.metadata["page_coords"] = page_coords
                if page_coords:
                    # 첫 페이지 좌표 (하위 호환)
                    doc.metadata["word_coords"] = page_coords[min(page_coords)]
                else:
                    # 좌표 추출 실패 시에도 인용 row(페이지 번호)를 유지해
                    # UI에서 페이지 점프만이라도 가능하게 한다 (크래시 방지).
                    doc.metadata.setdefault("citation_pages", pages)
                    doc.metadata["coord_extract_failed"] = True
