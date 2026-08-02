"""
문서 좌표 하이드레이션 모듈.
검색 결과 문서의 word_coords를 캐시에서 복구하거나 즉시 추출(Lazy)합니다.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import fitz
from langchain_core.documents import Document

from cache.coord_cache import coord_cache

logger = logging.getLogger(__name__)


def _extract_page_words_sync(
    path: str, page_num: int, chunk_bbox: Any | None
) -> list[Any] | None:
    """PDF 파일에서 특정 페이지의 단어 좌표를 동기로 추출합니다 (스레드 푸시용)."""
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
            for doc in docs_in_file:
                page = doc.metadata.get("page")
                if page is not None:
                    page_nums.append(page)
            coords_map = await coord_cache.get_coords_batch(file_hash, page_nums)

            for doc in docs_in_file:
                page_num = doc.metadata.get("page")
                if page_num is None:
                    continue

                coords = coords_map.get(page_num)

                if not coords:
                    logger.info(
                        f"[HYDRATE] 정밀 좌표 추출: {os.path.basename(path)} P{page_num}"
                    )
                    chunk_bbox = doc.metadata.get("bbox")
                    coords = await asyncio.to_thread(
                        _extract_page_words_sync, path, page_num, chunk_bbox
                    )
                    if coords:
                        await coord_cache.save_coords(file_hash, page_num, coords)

                if coords:
                    doc.metadata["word_coords"] = coords
