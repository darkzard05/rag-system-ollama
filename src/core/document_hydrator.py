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


async def hydrate_documents(docs: list[Document]) -> None:
    """문서 리스트의 좌표 데이터를 캐시에서 복구하거나, 없으면 즉시 추출(Lazy)합니다.

    파일 해시 그룹별로 캐시 미스 페이지를 모아 파일을 1회만 열어 일괄 추출하고
    (``_extract_pages_words_sync``), 추출 동시성은 세마포어로 제한한다.
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
        # D4: 그룹별 추출은 워커 스레드 1개의 단일 open 배치로 수행한다.
        # 호출 내 세마포어(루프 바인딩 회피를 위해 호출별 생성)로 동시 open을 제한.
        semaphore = asyncio.Semaphore(_HYDRATE_MAX_CONCURRENCY)
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

            # 캐시 미스 (page, bbox) 요청을 그룹 전체에서 1회씩만 수집한다.
            # 동일 페이지를 여러 청크가 참조해도 추출은 중복하지 않는다.
            # bbox가 청크마다 다르면 (page, bbox) 쌍 단위로 수집한다.
            miss_requests: list[tuple[int, Any | None]] = []
            seen_requests: set[str] = set()
            doc_bboxes: dict[int, Any | None] = {}
            for doc in docs_in_file:
                doc_id = id(doc)
                doc_bboxes[doc_id] = doc.metadata.get("bbox")
            for doc in docs_in_file:
                pages = doc.metadata.get("pages") or (
                    [doc.metadata["page"]]
                    if doc.metadata.get("page") is not None
                    else []
                )
                bbox = doc_bboxes[id(doc)]
                for page in pages:
                    if coords_map.get(page):
                        continue
                    key = f"{page}|{bbox!r}"
                    if key not in seen_requests:
                        seen_requests.add(key)
                        miss_requests.append((page, bbox))

            fresh_map: dict[str, list[Any] | None] = {}
            extracted: list[list[Any] | None] = []
            if miss_requests:
                logger.info(
                    f"[HYDRATE] 정밀 좌표 추출: {os.path.basename(path)} "
                    f"{len(miss_requests)}개 (page, bbox) 일괄 (단일 open)"
                )
                async with semaphore:
                    extracted = await asyncio.to_thread(
                        _extract_pages_words_sync, path, miss_requests
                    )
                for (page, bbox), coords in zip(miss_requests, extracted, strict=True):
                    fresh_map[f"{page}|{bbox!r}"] = coords

            # 추출(fresh) 성공 페이지만 병렬 저장한다. 캐시 적중 페이지는
            # 재저장하지 않는다 (기존 추출 데이터 보존 + 불필요한 I/O 방지).
            # 저장 실패는 하이드레이션 전체를 깨지 않고 로그-온리로 처리한다.
            save_targets = [
                (page, coords)
                for (page, _bbox), coords in zip(miss_requests, extracted, strict=True)
                if coords
            ]
            if save_targets:
                save_results = await asyncio.gather(
                    *(
                        coord_cache.save_coords(file_hash, page, coords)
                        for page, coords in save_targets
                    ),
                    return_exceptions=True,
                )
                for (page, _coords), outcome in zip(
                    save_targets, save_results, strict=True
                ):
                    if isinstance(outcome, BaseException):
                        logger.warning(
                            f"[HYDRATE] 좌표 저장 실패 ({os.path.basename(path)} "
                            f"P{page}): {outcome}"
                        )

            for doc in docs_in_file:
                pages = doc.metadata.get("pages") or (
                    [doc.metadata["page"]]
                    if doc.metadata.get("page") is not None
                    else []
                )
                page_coords: dict[int, list] = {}
                bbox = doc_bboxes[id(doc)]

                for page_num in pages:
                    coords = coords_map.get(page_num)
                    if coords:
                        page_coords[page_num] = coords
                        continue
                    fresh = fresh_map.get(f"{page_num}|{bbox!r}")
                    if fresh:
                        page_coords[page_num] = fresh

                doc.metadata["page_coords"] = page_coords
                if page_coords:
                    # 첫 페이지 좌표 (하위 호환)
                    doc.metadata["word_coords"] = page_coords[min(page_coords)]
                else:
                    # 좌표 추출 실패 시에도 인용 row(페이지 번호)를 유지해
                    # UI에서 페이지 점프만이라도 가능하게 한다 (크래시 방지).
                    doc.metadata.setdefault("citation_pages", pages)
                    doc.metadata["coord_extract_failed"] = True
