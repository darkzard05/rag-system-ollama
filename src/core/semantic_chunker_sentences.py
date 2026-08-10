"""
문장 분할 관심사 모듈 — ``SemanticChunkerSentencesMixin``.

라인 유형 분류(``_line_kind``), PDF 래핑 라인 리플로우(``_reflow_wrapped_lines``),
문장 분할(``_split_sentences``), 정제 문장 추가(``_add_cleaned_sentence``), 그리고
``split_text``의 짧은 문장 병합 단계(``_merge_short_sentences``)를 담당합니다.

[R2-10] ``semantic_chunker.py`` 모노리스에서 관심사별로 분리한 모듈입니다.
"""

import re
from typing import Any, Literal

from common.constants import ChunkingConstants


class SemanticChunkerSentencesMixin:
    """
    문장 분할 관심사 믹스인.

    아래 속성들은 ``EmbeddingBasedSemanticChunker.__init__``에서 설정되며,
    ``HEADER_PATTERN``은 breakpoints 모듈의 클래스 속성(MRO)을 사용합니다.
    """

    max_chunk_size: int
    _sentence_pattern: re.Pattern[str]
    HEADER_PATTERN: re.Pattern[str]

    # 축약어 마침표 패턴 — 문장 분할 시 가짜 경계로 취급하지 않아야 함 (R2-03)
    ABBREVIATION_PATTERN = re.compile(r"(?i)\b(?:et al|e\.g|i\.e|Dr|Mr|vs)\.(?=\s)")
    # 축약어 마침표 임시 보호용 센티널 (길이 1:1 치환 → 오프셋 보존)
    ABBREV_SENTINEL = "\x00"

    @staticmethod
    def _line_kind(line: str) -> Literal["blank", "structure", "list_marker", "text"]:
        """라인 유형 분류 — 리플로우 병합 판정에 사용됩니다."""
        stripped = line.strip()
        if not stripped:
            return "blank"
        if stripped.startswith("#") or stripped.startswith("|"):
            return "structure"
        if re.fullmatch(r"[-*_=]{3,}", stripped):
            return "structure"
        if re.match(r"^(?:[-*+]|\d+[.)])\s", stripped):
            return "list_marker"
        return "text"

    def _reflow_wrapped_lines(self, text: str) -> str:
        """PDF 마크다운의 래핑 줄바꿈을 공백으로 병합하는 리플로우 전처리.

        개행(\\n)을 정확히 1문자(공백)로 치환하므로 텍스트 전체 길이와
        비-개행 문자의 위치가 보존되어, 산출된 문장 오프셋(start/end)이 원본
        텍스트 좌표와 그대로 일치합니다 (좌표 하이드레이션 무결성 유지).

        병합 규칙: (1) 연속된 일반 텍스트 라인은 병합, (2) 리스트 마커 라인은
        자신의 래핑 연속 라인과 병합. 빈 줄/헤더/테이블 행/수평선은 구조 요소로
        병합하지 않고, 일반 텍스트 뒤의 새 리스트/헤더 시작도 병합하지 않습니다.
        """
        out: list[str] = []
        prev_kind: Literal["blank", "structure", "list_marker", "text"] = "blank"
        for line in text.split("\n"):
            kind = self._line_kind(line)
            if out and prev_kind in ("text", "list_marker") and kind == "text":
                out.append(" ")
            elif out:
                out.append("\n")
            out.append(line)
            prev_kind = kind
        return "".join(out)

    def _split_sentences(self, text: str) -> list[dict]:
        """
        문장 단위로 텍스트를 분할하고 오프셋 정보를 함께 반환합니다.
        [최적화] 마크다운 테이블 행(|)을 보호하며, 너무 긴 문장은 강제로 분할합니다.
        [R2-03] 분할 전에 래핑 라인을 리플로우(개행→공백)하여 PDF 줄바꿈 파편이
        독립 "문장"으로 남지 않도록 합니다.
        """
        # [R2-03] 문장 분할 전 리플로우 — 오프셋 보존(개행 1:1 → 공백)
        text = self._reflow_wrapped_lines(text)
        lines = text.split("\n")
        temp_sentences: list[dict[str, Any]] = []
        current_pos = 0

        # 1. 행 단위로 훑으며 테이블 보호 및 문장 분할
        for line in lines:
            line_len = len(line)
            stripped = line.strip()

            # 테이블 행 감지 ( | 로 시작하고 | 로 끝나는 패턴)
            if stripped.startswith("|") and stripped.endswith("|"):
                self._add_cleaned_sentence(
                    temp_sentences, line, current_pos, current_pos + line_len
                )
            else:
                # [R2-03] 축약어 마침표를 센티널로 보호해 가짜 문장 경계 차단.
                # 센티널은 길이 1:1 치환이므로 정규식 매치 위치는 원본 라인 좌표와 동일.
                protected_line = self.ABBREVIATION_PATTERN.sub(
                    lambda m: m.group(0)[:-1] + self.ABBREV_SENTINEL, line
                )
                # 일반 텍스트는 정규식으로 분할 (원본 라인에서 슬라이스 → 마침표 복원 불필요)
                parts = list(self._sentence_pattern.finditer(protected_line))
                last_pos = 0
                for match in parts:
                    sep_end = match.end()
                    segment_text = line[last_pos:sep_end]
                    if segment_text.strip():
                        self._add_cleaned_sentence(
                            temp_sentences,
                            segment_text,
                            current_pos + last_pos,
                            current_pos + sep_end,
                        )
                    last_pos = sep_end

                if last_pos < len(line):
                    remaining_text = line[last_pos:]
                    if remaining_text.strip():
                        self._add_cleaned_sentence(
                            temp_sentences,
                            remaining_text,
                            current_pos + last_pos,
                            current_pos + len(line),
                        )

            current_pos += line_len + 1  # \n 포함

        # 2. 너무 긴 세그먼트 강제 분할 (OOM 방지 및 하드 스플릿 플래그 처리)
        hard_split_limit: int = ChunkingConstants.MAX_HARD_SPLIT_LEN.value
        if self.max_chunk_size > 0:
            hard_split_limit = min(
                ChunkingConstants.MAX_HARD_SPLIT_LEN.value,
                int(self.max_chunk_size * 1.5),
            )

        final_sentences: list[dict[str, Any]] = []
        for seg in temp_sentences:
            seg_text = str(seg["text"])
            seg_start = int(seg["start"])

            if len(seg_text) <= hard_split_limit:
                final_sentences.append(seg)
            else:
                # 강제 분할 로직
                curr_pos = 0
                while curr_pos < len(seg_text):
                    sub_len: int = int(hard_split_limit)
                    if curr_pos + sub_len < len(seg_text):
                        # 공백 기준으로 끊기 시도
                        last_space = seg_text.rfind(" ", curr_pos, curr_pos + sub_len)
                        if last_space != -1 and last_space > curr_pos + (sub_len // 2):
                            sub_len = int(last_space - curr_pos + 1)

                    sub_text = seg_text[curr_pos : curr_pos + sub_len]
                    is_last_sub = curr_pos + sub_len >= len(seg_text)

                    self._add_cleaned_sentence(
                        final_sentences,
                        sub_text,
                        seg_start + curr_pos,
                        seg_start + curr_pos + len(sub_text),
                        is_hard_split=not is_last_sub,
                    )
                    curr_pos += sub_len

        return final_sentences

    def _add_cleaned_sentence(
        self,
        target_list: list[dict[str, Any]],
        raw_text: str,
        start_offset: int,
        end_offset: int,
        is_hard_split: bool = False,
    ):
        """정제된 문장을 리스트에 추가합니다."""
        l_stripped = raw_text.lstrip()
        n_leading = len(raw_text) - len(l_stripped)

        stripped = l_stripped.rstrip()
        n_trailing = len(l_stripped) - len(stripped)

        final_start = start_offset + n_leading
        final_end = end_offset - n_trailing

        embed_text = stripped.replace("\n", " ")
        if embed_text:
            target_list.append(
                {
                    "text": embed_text,
                    "start": final_start,
                    "end": final_end,
                    "is_hard_split": is_hard_split,
                }
            )

    def _merge_short_sentences(
        self, raw_sentences: list[dict]
    ) -> tuple[list[dict], set[int]]:
        """너무 짧은 문장을 병합하고 헤더 위치 집합을 캐싱해 반환합니다.

        [최적화] 병합 전 1회만 헤더 여부를 계산하여 캐싱하며, 헤더 감지 정규식은
        클래스 상수 ``HEADER_PATTERN``(# 패턴 또는 대문자 시작 섹션)을 사용합니다.
        반환값은 ``(병합된 문장 리스트, 헤더 인덱스 집합)``입니다.
        """
        min_merge_len = ChunkingConstants.MIN_MERGE_LEN.value

        sentences = []
        if raw_sentences:
            # [최적화] 병합 전 1회만 헤더 여부를 계산하여 캐싱
            for raw_s in raw_sentences:
                raw_s["is_header"] = bool(
                    self.HEADER_PATTERN.match(raw_s["text"].strip())
                )

            current_s = raw_sentences[0]
            for s in raw_sentences[1:]:
                # [수정] 현재 문장이 헤더거나, 다음 문장이 헤더인 경우 병합 제외 (Clean Section Preservation)
                # [최적화] 미리 캐싱된 is_header 값 재사용 (정규식 중복 호출 제거)
                is_curr_header = current_s.get("is_header", False)
                is_next_header = s.get("is_header", False)

                can_merge = (
                    not is_curr_header  # ✅ 현재 문장이 헤더면 절대 합치지 않음
                    and not is_next_header  # 다음 문장이 헤더면 합치지 않음
                    and not current_s.get("is_hard_split", False)
                    and (len(s["text"]) < min_merge_len)
                    and (len(current_s["text"]) + len(s["text"]) < 1000)
                )

                if can_merge:
                    current_s["text"] += " " + s["text"]
                    current_s["end"] = s["end"]
                    current_s["is_hard_split"] = s.get("is_hard_split", False)
                    # 병합된 문장은 헤더가 아님 (텍스트가 변경되었으므로)
                    current_s["is_header"] = False
                else:
                    sentences.append(current_s)
                    current_s = s
            sentences.append(current_s)

        # [최적화] 헤더 위치 미리 캐싱 (O(1) lookup용) — 이미 위에서 계산 완료
        header_indices = {
            i for i, s in enumerate(sentences) if s.get("is_header", False)
        }
        return sentences, header_indices
