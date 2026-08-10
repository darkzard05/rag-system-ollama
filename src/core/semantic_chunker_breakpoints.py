"""
브레이크포인트 탐색 관심사 모듈 — ``SemanticChunkerBreakpointsMixin``.

유사도 거리 분포에서 분할 지점을 찾는 ``_find_breakpoints``와 헤더 감지용
정규식 상수(``HEADER_PATTERN``, ``HEADER_SIMPLE``)를 담당합니다.

[R2-10] ``semantic_chunker.py`` 모노리스에서 관심사별로 분리한 모듈입니다.
"""

import logging
import re
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


class SemanticChunkerBreakpointsMixin:
    """
    브레이크포인트 탐색 관심사 믹스인.

    아래 속성들은 ``EmbeddingBasedSemanticChunker.__init__``에서 설정됩니다.
    """

    breakpoint_threshold_type: str
    breakpoint_threshold_value: float
    similarity_threshold: float
    max_chunk_size: int

    # 복합 헤더 패턴 (Markdown # + 대문자 섹션명)
    HEADER_PATTERN = re.compile(
        r"^(\s*#{1,6}\s+.+|\d+\s+[A-Z]{2,}.*|[A-Z]{3,}(\s+[A-Z]{3,})*)$",
        re.MULTILINE,
    )
    # 간단 헤더 패턴 (breakpoint 탐지용)
    HEADER_SIMPLE = re.compile(r"^\s*#{1,6}\s+.+", re.MULTILINE)

    def _find_breakpoints(
        self, distances: list[float], sentences: list[dict] | None = None
    ) -> list[int]:
        """
        유사도 거리(1-cos_sim) 분포를 분석하여 분할 지점을 찾습니다.
        [고도화] 마크다운 헤더 감지 시 강제 분할 지점으로 추가합니다.
        """
        if not distances:
            return []

        dist_array = np.array(distances)
        threshold = 0.0

        if self.breakpoint_threshold_type == "percentile":
            threshold = float(
                np.percentile(dist_array, self.breakpoint_threshold_value)
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            threshold = float(
                np.mean(dist_array)
                + self.breakpoint_threshold_value * np.std(dist_array)
            )
        elif self.breakpoint_threshold_type == "interquartile":
            # [수정] Mypy 타입 추론 지원을 위해 리스트 형태의 인덱스 전달 시 반환값을 명시적으로 처리
            percentiles = cast(np.ndarray, np.percentile(dist_array, [25, 75]))
            q1, q3 = float(percentiles[0]), float(percentiles[1])
            iqr = q3 - q1
            threshold = float(
                np.mean(dist_array) + self.breakpoint_threshold_value * iqr
            )
        elif self.breakpoint_threshold_type == "gradient":
            # 거리가 급격히 변하는 지점 감지
            threshold = float(
                np.percentile(np.gradient(dist_array), self.breakpoint_threshold_value)
            )
        elif self.breakpoint_threshold_type == "similarity_threshold":
            # [R2-01] config 기본 모드 — 명시 분기 (기존 else 암묵 폴백 제거)
            threshold = float(self.similarity_threshold)
        else:
            threshold = float(self.similarity_threshold)  # 알 수 없는 유형 방어용
            logger.warning(
                f"[Chunker] 알 수 없는 breakpoint_threshold_type="
                f"{self.breakpoint_threshold_type!r} — similarity_threshold 폴백 사용"
            )

        # 1. 유사도 기반 분기점 추출
        breakpoints = (np.where(dist_array > threshold)[0] + 1).tolist()

        # 2. [고도화] 마크다운 헤더 감지 기반 강제 분할
        if sentences:
            header_bps = []
            for i, s in enumerate(sentences):
                # 문장의 시작이 헤더 패턴인 경우 (첫 문장 제외)
                if i > 0 and self.HEADER_SIMPLE.match(s["text"].strip()):
                    header_bps.append(i)

            if header_bps:
                logger.info(
                    f"[Chunker] {len(header_bps)}개의 마크다운 헤더를 감지하여 강제 분할 지점으로 설정합니다."
                )
                breakpoints = sorted(set(breakpoints + header_bps))

        # 3. [안전 장치] 너무 긴 청크 방지
        if sentences:
            current_len = 0
            safety_bps = []
            for i, s in enumerate(sentences):
                current_len += len(s["text"])
                if current_len > (self.max_chunk_size * 0.9):
                    # 이미 breakpoints에 이 근처 지점이 있는지 확인
                    if not any(abs(bp - (i + 1)) <= 1 for bp in breakpoints):
                        safety_bps.append(i + 1)
                    current_len = 0
            if safety_bps:
                breakpoints = sorted(set(breakpoints + safety_bps))

        return breakpoints
