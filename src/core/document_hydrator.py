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

# D4: 하이드레이션 동시성 상한. 파일 그룹별 배치 추출 1건을 워커 스레드 1개에
# 매핑하므로, 이 세마포어는 동시 open 중인 PDF 핸들 수(= 워커 스레드 수)를 제한한다.
_HYDRATE_MAX_CONCURRENCY = 4


def _extract_pages_words_sync(
    path: str, requests: list[tuple[int, Any | None]]
) -> list[list[Any] | None]:
    """여러 페이지의 단어 좌표를 파일을 1회만 열어 일괄 추출합니다 (스레드 푸시용).

    ``requests``는 ``(page_num, chunk_bbox)`` 순서 리스트이며, 반환도 동일 순서의
    ``coords | None`` 리스트다. 동일 페이지가 bbox만 달리해 중복 요청돼도 open은
    1회이며 페이지 읽기만 반복한다. bbox 추출이 비면 전체 페이지로 폴백하고,
    그래도 비면 ``None``을 반환해 호출부가 좌표なし로 남기게 한다.
    """
    import pymupdf as fitz  # lazy: Rect/예외 타입용으로만 사용

    from common.pdf_utils import open_pdf_document  # 공용 PDF 오픈 퍼사드 (R: Group 7)

    results: list[list[Any] | None] = [None] * len(requests)
    try:
        with open_pdf_document(path) as doc_obj:
            for i, (page_num, chunk_bbox) in enumerate(requests):
                try:
                    page_obj = doc_obj[page_num - 1]
                except IndexError:
                    logger.warning(f"[HYDRATE] 페이지 인덱스 초과: P{page_num}")
                    continue
                try:
                    if chunk_bbox:
                        raw_words = page_obj.get_text(
                            "words", clip=fitz.Rect(chunk_bbox)
                        )
                    else:
                        raw_words = page_obj.get_text("words")
                    coords = [(w[0], w[1], w[2], w[3], w[4]) for w in raw_words]
                    # Fallback: bbox-scoped extract failed -> retry whole page
                    if not coords and chunk_bbox is not None:
                        raw_words = page_obj.get_text("words")
                        coords = [(w[0], w[1], w[2], w[3], w[4]) for w in raw_words]
                    results[i] = coords or None
                except (OSError, fitz.FileDataError) as e:
                    logger.error(
                        f"[HYDRATE] 파일 처리 중 오류 ({os.path.basename(path)}): {e}"
                    )
    except (OSError, Exception) as e:
        logger.error(f"[HYDRATE] 파일 열기 실패 ({os.path.basename(path)}): {e}")
    return results


def _extract_page_words_sync(
    path: str, page_num: int, chunk_bbox: Any | None
) -> list[Any] | None:
    """PDF 파일에서 특정 페이지의 단어 좌표를 동기로 추출합니다 (스레드 푸시용).

    단일 페이지용 호환 래퍼 — 내부는 단일 오픈 배치 경로를 재사용한다.
    """
    return _extract_pages_words_sync(path, [(page_num, chunk_bbox)])[0]


def _get_doc_pages(doc: Document) -> list[int]:
    """문서 메타데이터에서 페이지 번호 목록 추출 헬퍼."""
    pages = doc.metadata.get("pages")
    if pages:
        return list(pages)
    page = doc.metadata.get("page")
    return [page] if page is not None else []


async def hydrate_documents(docs: list[Document]) -> None:
    """문서 리스트의 좌표 데이터를 캐시에서 복구하거나, 없으면 즉시 추출(Lazy)합니다.

    파일 해시 그룹별로 캐시 미스 페이지를 모아 파일을 1회만 열어 일괄 추출하고
    (``_extract_pages_words_sync``), 추출 동시성은 세마포어로 제한한다.
    """
    # 1. 파일 경로 및 해시 기준으로 단일 패스 그룹화 (경로 존재 여부는 캐시)
    file_groups: dict[tuple[str, str], list[Document]] = {}
    path_exists_cache: dict[str, bool] = {}

    for doc in docs:
        if "word_coords" in doc.metadata or not doc.metadata.get("has_coordinates"):
            continue
        path = doc.metadata.get("file_path")
        file_hash = doc.metadata.get("file_hash")
        if not (path and file_hash):
            continue
        if path not in path_exists_cache:
            path_exists_cache[path] = os.path.exists(path)
        if path_exists_cache[path]:
            file_groups.setdefault((path, file_hash), []).append(doc)

    if not file_groups:
        return

    semaphore = asyncio.Semaphore(_HYDRATE_MAX_CONCURRENCY)

    async def _hydrate_file_group(
        path: str, file_hash: str, docs_in_file: list[Document]
    ) -> None:
        # doc별 pages 조회 결과를 1회만 계산하여 재사용
        doc_pages_map = {id(doc): _get_doc_pages(doc) for doc in docs_in_file}
        page_nums = list(
            dict.fromkeys(p for pages in doc_pages_map.values() for p in pages)
        )

        try:
            coords_map = await coord_cache.get_coords_batch(file_hash, page_nums)
        except CoordCacheReadError as e:
            logger.warning(
                f"[HYDRATE] 좌표 캐시 읽기 실패 ({os.path.basename(path)}, file_hash={file_hash}): {e}"
            )
            for doc in docs_in_file:
                doc.metadata["coord_cache_error"] = True
            return

        # 캐시 미스 페이지는 항상 전체 페이지(bbox=None)로 일괄 추출
        miss_pages = [p for p in page_nums if not coords_map.get(p)]
        fresh_coords: dict[int, list[Any]] = {}
        if miss_pages:
            collapsed = [(p, None) for p in miss_pages]
            logger.info(
                f"[HYDRATE] 정밀 좌표 추출: {os.path.basename(path)} {len(collapsed)}개 (단일 open)"
            )
            async with semaphore:
                extracted = await asyncio.to_thread(
                    _extract_pages_words_sync, path, collapsed
                )
            fresh_coords = {
                p: c for (p, _), c in zip(collapsed, extracted, strict=True) if c
            }

            # 신규 추출 성공 페이지만 비동기 배치 저장
            if fresh_coords:
                save_results = await asyncio.gather(
                    *(
                        coord_cache.save_coords(file_hash, p, c)
                        for p, c in fresh_coords.items()
                    ),
                    return_exceptions=True,
                )
                for p, outcome in zip(fresh_coords.keys(), save_results, strict=True):
                    if isinstance(outcome, BaseException):
                        logger.warning(
                            f"[HYDRATE] 좌표 저장 실패 ({os.path.basename(path)} P{p}): {outcome}"
                        )

        # 단일 병합 딕셔너리로 O(1) 매핑 및 중복 조회 제거
        all_coords = coords_map | fresh_coords
        for doc in docs_in_file:
            pages = doc_pages_map[id(doc)]
            page_coords = {p: all_coords[p] for p in pages if p in all_coords}
            doc.metadata["page_coords"] = page_coords
            if page_coords:
                doc.metadata["word_coords"] = page_coords[min(page_coords)]
            else:
                doc.metadata.setdefault("citation_pages", pages)
                doc.metadata["coord_extract_failed"] = True

    # 각 파일 그룹은 독립적인 PDF 경로/캐시 키를 가지므로 동시 실행
    await asyncio.gather(
        *(
            _hydrate_file_group(path, file_hash, docs)
            for (path, file_hash), docs in file_groups.items()
        )
    )
