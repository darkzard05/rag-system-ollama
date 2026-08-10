"""
메타데이터 역매핑 관심사 모듈 — ``SemanticChunkerMetadataMixin``.

청크 오프셋을 원본 문서 범위에 역매핑해 페이지/섹션 메타데이터를 산출하는
``_extract_metadata_for_chunk``를 담당합니다.

[R2-10] ``semantic_chunker.py`` 모노리스에서 관심사별로 분리한 모듈입니다.
"""

from typing import Any, cast


class SemanticChunkerMetadataMixin:
    """
    메타데이터 역매핑 관심사 믹스인.

    ``split_documents``의 ``split_text`` 결과 청크에 대해 원본 ``doc_ranges``
    기반으로 페이지·섹션 정보를 병합합니다.
    """

    def _extract_metadata_for_chunk(
        self, chunk: dict[str, Any], doc_ranges: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        청크에 대한 메타데이터를 추출하고 병합합니다.
        """
        c_start = chunk["start"]
        c_end = chunk["end"]

        overlapping_pages = []
        merged_metadata: dict[str, Any] = {}

        for doc_range in doc_ranges:
            # 범위 겹침 확인: [start1, end1] 과 [start2, end2]
            if max(c_start, doc_range["start"]) < min(c_end, doc_range["end"]):
                # [수정] Mypy 타입 추론 강화를 위해 cast 사용
                metadata = cast(dict[str, Any], doc_range["metadata"])
                page = metadata.get("page")
                if page and page not in overlapping_pages:
                    overlapping_pages.append(page)

                # 기본 메타데이터는 첫 번째 겹치는 문서에서 가져오되, 페이지 정보는 업데이트
                if not merged_metadata:
                    merged_metadata = metadata.copy()

        if overlapping_pages:
            merged_metadata["pages"] = sorted(overlapping_pages)
            # 하위 호환성을 위해 단일 page 필드는 첫 페이지로 유지
            merged_metadata["page"] = overlapping_pages[0]
            # 다중 페이지 여부 표시
            merged_metadata["is_cross_page"] = len(overlapping_pages) > 1

        # [핵심 수정] 추출된 섹션 정보를 메타데이터에 주입
        merged_metadata["current_section"] = chunk.get("current_section", "일반 본문")

        merged_metadata["start_index"] = c_start
        merged_metadata["end_index"] = c_end

        return merged_metadata
