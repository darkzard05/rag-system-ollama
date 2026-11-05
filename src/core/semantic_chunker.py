"""
임베딩 기반 의미론적 텍스트 분할기 — 단일 파일 통합 모듈.

이 모듈은 문서를 문장 단위로 우선 분할한 후, 인접 문장 간의 임베딩 유사도를
계산하여 유사도가 낮은 지점을 경계로 선택합니다. 이를 통해 의미론적으로
일관성 있는 청크를 생성합니다.

5개 믹스인 계층을 제거하고 EmbeddingBasedSemanticChunker 단일 클래스로 평탄화되었습니다.
공개 API(``EmbeddingBasedSemanticChunker.split_documents``·``_split_sentences``·
``__init__`` 시그니처)는 기존과 동일하게 유지되어 외부 import 경로가 그대로 동작합니다.
"""

import asyncio
import bisect
import logging
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xxhash
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import MODEL_CACHE_DIR, is_test_env
from common.constants import ChunkingConstants
from common.similarity import cosine_similarity
from core.model_loader import is_ollama_embedder
from services.optimization.caching_optimizer import CacheManager

logger = logging.getLogger(__name__)

# Backward-compatible alias: older tests import this module-local name.
_is_ollama_embedder = is_ollama_embedder


# [상수] 표준 섹션 키워드 (정제 시 활용)
STANDARD_SECTION_KEYWORDS = [
    "ABSTRACT",
    "INTRODUCTION",
    "RELATED WORK",
    "METHOD",
    "EXPERIMENT",
    "RESULT",
    "CONCLUSION",
    "REFERENCE",
    "ACKNOWLEDGMENT",
]


class EmbeddingBasedSemanticChunker:
    """
    임베딩 기반 의미론적 텍스트 분할기 (단일 클래스 평탄화 완료).

    문장 단위로 분할 후, 임베딩 유사도 기반으로 의미 경계를 탐지하여
    일관성 있는 청크를 생성합니다.
    """

    # 토큰 예산 그룹핑 시 문장 결합/구두점 병합으로 인한 미세 오차 여유.
    TOKEN_GROUP_MARGIN = 8

    # 축약어 마침표 패턴 — 문장 분할 시 가짜 경계로 취급하지 않아야 함 (R2-03)
    ABBREVIATION_PATTERN = re.compile(r"(?i)\b(?:et al|e\.g|i\.e|Dr|Mr|vs)\.(?=\s)")
    # 축약어 마침표 임시 보호용 센티널 (길이 1:1 치환 → 오프셋 보존)
    ABBREV_SENTINEL = "\x00"

    # 복합 헤더 패턴 (Markdown # + 대문자 섹션명)
    HEADER_PATTERN = re.compile(
        r"^(\s*#{1,6}\s+.+|\d+\s+[A-Z]{2,}.*|[A-Z]{3,}(\s+[A-Z]{3,})*)$",
        re.MULTILINE,
    )
    # 간단 헤더 패턴 (breakpoint 탐지용)
    HEADER_SIMPLE = re.compile(r"^\s*#{1,6}\s+.+", re.MULTILINE)

    def __init__(
        self,
        embedder: Embeddings,
        buffer_size: int = 1,  # [추가] 문맥 윈도우 크기
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_value: float = 95.0,
        sentence_split_regex: str = r"(?<=[.?!])\s+",  # [개선] Lookbehind 적용
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        chunk_overlap: int = 1,  # [추가] 겹칠 문장 수 (Context preservation)
        similarity_threshold: float = 0.6,
        batch_size: int = 64,
        cache_manager: CacheManager | None = None,
        max_chunk_tokens: int | None = None,
        token_count_fn: Callable[[list[str]], list[int]] | None = None,
    ):
        """
        의미론적 청킹 분할기를 초기화합니다.
        """
        self.embedder = embedder
        self.buffer_size = buffer_size  # [추가]
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_value = breakpoint_threshold_value
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap  # [추가]
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        # [P0] 임베딩 모델의 실제 토큰 상한. 문자 수(max_chunk_size)만으로는
        # 한국어에서 448토큰을 크게 넘기므로, 문장 벡터를 유지한 채 토큰
        # 예산으로 그룹핑/병합을 제한한다.
        self.max_chunk_tokens = max_chunk_tokens
        self._token_count_fn = token_count_fn

        # [최적화] 청킹 임베딩 전용 캐시 관리자 설정.
        # 메모리 레이어(L1)로 동일 프로세스 내 동일 문장의 중복 임베딩을
        # 방지한다. 디스크 레이어(L3)는 캐시 매니저에 구성해 두되, 청킹 hot
        # path(``_get_embeddings``)는 [최적화-04]에 따라 persist_to_disk=False로
        # 호출하여 문장마다 개별 파일을 쓰지 않는다 (청크 벡터는 VectorStoreCache가
        # index.faiss/doc_splits.json으로 일괄 영속화). 희귀한 차원 불일치 복구
        # 경로(``_reembed_single``)만 보정 벡터를 디스크에 남긴다.
        # 디스크 경로는 전역 MODEL_CACHE_DIR 하위의 embedding_cache 로 격리해
        # 기타 응답 캐시와 혼재되지 않게 한다.
        #
        # [안전장치] 테스트 환경(IS_UNIT_TEST/IS_CI_TEST)에서는 가짜 임베더
        # (FakeEmbeddings 등)가 생성한 벡터가 프로덕션 캐시 공간을 오염하지
        # 않도록 디스크 캐시 경로를 OS 임시 디렉토리로 분리한다. 가짜 임베더는
        # 실제 모델과 출력 차원이 다를 수 있어(1536 vs 768), 프로덕션 경로에
        # 기록되면 이후 chunker가 그 벡터를 히트해 FAISS 인덱스-쿼리 차원
        # 불일치(assert d == self.d)를 유발했다.
        _is_test_env = is_test_env()
        _embedding_cache_dir = (
            str(Path(tempfile.gettempdir()) / "embedding_cache_test")
            if _is_test_env
            else str(Path(MODEL_CACHE_DIR) / "embedding_cache")
        )
        self.cache_manager = cache_manager or CacheManager(
            enable_memory_cache=True,
            enable_semantic_cache=False,
            enable_disk_cache=False,
        )

        # [최적화] 모델 식별을 위한 이름 추출 (Ollama 및 HuggingFace 지원 강화)
        # 모델명은 항상 문자열이어야 하므로 (emb:{model}:{hash} 캐시 키) cast로 보장.
        self.model_name = cast(
            str,
            getattr(embedder, "model", None)
            or getattr(embedder, "model_name", "default_model"),
        )

        # ✅ 정규식 사전 컴파일 (성능 최적화)
        self._sentence_pattern = re.compile(self.sentence_split_regex)

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

    async def _get_embeddings(
        self, texts: list[str], normalize: bool = False
    ) -> np.ndarray:
        """
        텍스트 리스트의 임베딩을 생성합니다 (배치 처리 및 캐싱 강화).
        [수정] 차원 불일치/None 캐시 벡터는 제거하지 않고 해당 텍스트를
        재임베딩하여 복구하며, 반환 행렬의 행 수는 항상 입력 텍스트 수와 동일합니다.
        """
        if not texts:
            return np.array([]).reshape(0, 0)

        all_results: list[np.ndarray | None] = [None] * len(texts)
        missing_texts: list[str] = []
        text_to_output_indices: dict[str, list[int]] = {}
        newly_embedded: list[tuple[int, str, np.ndarray]] = []

        # 1. 정제 및 캐시 확인 (중복 제거 후 gather 병렬 조회)
        norm_texts: list[str] = [" ".join(text.split()) for text in texts]
        norm_to_indices: dict[str, list[int]] = {}
        unique_norms: list[str] = []
        for i, norm_text in enumerate(norm_texts):
            if norm_text not in norm_to_indices:
                norm_to_indices[norm_text] = []
                unique_norms.append(norm_text)
            norm_to_indices[norm_text].append(i)
        norm_to_key: dict[str, str] = {
            norm_text: f"emb:{self.model_name}:{xxhash.xxh64(norm_text.encode()).hexdigest()}"
            for norm_text in unique_norms
        }

        cached_raws = await asyncio.gather(
            *(
                self.cache_manager.get(norm_to_key[norm_text])
                for norm_text in unique_norms
            )
        )

        for norm_text, cached_raw in zip(unique_norms, cached_raws, strict=True):
            if cached_raw is not None:
                cached_vec = (
                    cached_raw["vector"]
                    if isinstance(cached_raw, dict) and "vector" in cached_raw
                    else cached_raw
                )
                cached_vector = self._as_valid_vector(cached_vec)
                if cached_vector is not None:
                    for i in norm_to_indices[norm_text]:
                        all_results[i] = cached_vector
                    continue

            missing_texts.append(norm_text)
            text_to_output_indices[norm_text] = norm_to_indices[norm_text]

        # 2. 누락분 배치 임베딩 수행 (캐시 저장은 모든 배치 성공 후 단일 패스)
        if missing_texts:
            hit_rate = 1.0 - len(missing_texts) / len(texts)
            logger.debug(
                f"[Chunker] {len(missing_texts)}개 문장 신규 임베딩 생성 중 (Batch Size: {self.batch_size}, hit율: {hit_rate:.1%})..."
            )

            # [최적화] Ollama 임베더는 embed_documents가 입력 전체를 단일 HTTP
            # 요청으로 전송하므로, 누락분을 한 번의 embed_documents 호출로 모두
            # 묶어 HTTP 왕복 횟수를 1로 고정한다. (배치 사이즈로 쪼개면 왕복만
            # 늘어나 성능이 떨어짐) 비-Ollama(HuggingFace 등)는 기존처럼
            # batch_size 단위로 분할하여 메모리/디바이스 제약을 존중한다.
            if is_ollama_embedder(self.embedder):
                embed_batches: list[list[str]] = [missing_texts]
            else:
                embed_batches = [
                    missing_texts[b : b + self.batch_size]
                    for b in range(0, len(missing_texts), self.batch_size)
                ]

            # [수정] 배치 루프 동안 메모리에만 수집, 캐시 저장은 루프 이후 단일 패스로 수행
            for b_idx, batch in enumerate(embed_batches):
                # 배치 텍스트 → 원본 출력 인덱스 역매핑 (중복 텍스트가 섞여도
                # 슬라이싱 어긋남 없이 실패 배치를 정확히 추적한다).
                batch_indices = [
                    idx for t in batch for idx in text_to_output_indices.get(t, [])
                ]

                try:
                    # [최적화] 동기 임베딩 생성을 비동기 스레드로 분리 (embed 동안 모델 pin)
                    from core.resource_manager import get_resource_manager

                    coordinator = get_resource_manager()
                    async with coordinator.use_embedder(embedder=self.embedder):
                        batch_vecs = await asyncio.to_thread(
                            self.embedder.embed_documents, batch
                        )

                    for batch_text, vec in zip(batch, batch_vecs, strict=False):
                        vec_np = np.array(vec, dtype="float32")
                        for output_idx in text_to_output_indices.get(batch_text, []):
                            all_results[output_idx] = vec_np
                            newly_embedded.append(
                                (output_idx, texts[output_idx], vec_np)
                            )

                except Exception as e:
                    logger.error(
                        f"[Chunker] 배치 {b_idx} 임베딩 생성 중 오류 발생: {e}",
                        exc_info=True,
                    )
                    # 배치 실패 시 해당 인덱스의 결과 제거
                    for idx in batch_indices:
                        all_results[idx] = None
                    raise

            # [수정] 모든 배치 성공 후 단일 wait_for + gather 병렬 배치 저장
            # (逐차 per-item wait_for는 miss가 많을수록 t_embed 지배 → 1회로 축소)
            # P1: CacheBackedEmbeddings 패턴 — 저장 키도 조회와 동일하게
            # 정규화(" ".join(split))한다. 기존에는 조회는 정규화 키,
            # 저장은 원문 키라 공백 차이만으로 영구 miss가 발생했다.
            #
            # [최적화-04] 청킹 중 문장 단위 임베딩은 L1 메모리 캐시에만 저장한다.
            # persist_to_disk=True는 문장마다 개별 .cache/.cache.meta 파일을 쓰고
            # security_manager.full_verification(SHA-256 + HMAC)을 유발해, 20p PDF
            # 한 건당 수백 회의 디스크 I/O 및 해시 계산을 asyncio.to_thread로
            # 폭발시켰다. 최종 결과물은 VectorStoreCache가 index.faiss/
            # doc_splits.json으로 원자적 일괄 영속화하므로(동일 PDF 재업로드 시
            # file_hash 캐시 히트) 문장 임베딩의 중복 디스크 저장은 불필요하다.
            try:
                unique_new = {
                    " ".join(text.split()): vec for _, text, vec in newly_embedded
                }
                await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            self.cache_manager.set(
                                f"emb:{self.model_name}:{xxhash.xxh64(t.encode()).hexdigest()}",
                                {"vector": v.tolist(), "cache_version": "1.0"},
                                persist_to_disk=False,
                            )
                            for t, v in unique_new.items()
                        )
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.error("[Chunker] 캐시 저장 타임아웃 (일부 항목 미저장 가능)")
                raise

        # 3. [수정] 차원 불일치/누락 벡터는 개별 재임베딩으로 복구 (제거하지 않음)
        expected_dim = self._resolve_expected_dim(all_results, newly_embedded)
        if expected_dim is None:
            raise ValueError(
                "[Chunker] 임베딩 결과가 하나도 없어 벡터 행렬을 구성할 수 없습니다."
            )

        for i, res_vec in enumerate(all_results):
            if res_vec is None or int(res_vec.shape[0]) != expected_dim:
                logger.warning(
                    f"[Chunker] 캐시된 벡터 차원 불일치 감지 (Index: {i}, "
                    f"Dim: {None if res_vec is None else res_vec.shape[0]} "
                    f"!= Expected: {expected_dim}). 해당 문장을 재임베딩하여 복구합니다."
                )
                all_results[i] = await self._reembed_single(texts[i], expected_dim)

        # 4. [수정] 행렬 조립 — 행 수는 항상 입력 텍스트 수와 동일해야 함
        valid_vectors = [cast(np.ndarray, v) for v in all_results]
        if len(valid_vectors) != len(texts):
            raise ValueError(
                f"[Chunker] 임베딩 행렬의 행 수({len(valid_vectors)})가 "
                f"입력 텍스트 수({len(texts)})와 일치하지 않습니다."
            )

        embeddings_matrix = np.stack(valid_vectors).astype("float32")
        if normalize:
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            embeddings_matrix = np.divide(
                embeddings_matrix,
                norms,
                out=np.zeros_like(embeddings_matrix),
                where=norms > 1e-9,
            )
        return embeddings_matrix

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """외부 텍스트 리스트를 캐시·배치 공유 경로로 임베딩한다 (raw, 비정규화).

        오버플로우 청킹(``core.chunking``)이 문장 텍스트와 정상 청크 텍스트를
        합쳐 단일 ``embed_documents`` 왕복으로 처리하기 위한 진입점이다.
        캐시 조회·정제·중복 제거·Ollama 단일 호출·HF 배치 분할은
        ``_get_embeddings``과 동일하며, 반환 행렬의 행 수는 항상 입력 텍스트 수와
        동일하다. 문장 벡터로 쓸 때는 호출자가 ``l2_normalize``로 정규화한다
        (R2-02 단위노름 배선).
        """
        return await self._get_embeddings(texts, normalize=False)

    @staticmethod
    def l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """행 단위 L2 정규화 — ``_get_embeddings(normalize=True)``와 동일 연산."""
        arr = np.asarray(matrix, dtype="float32")
        if arr.size == 0:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.divide(
            arr,
            norms,
            out=np.zeros_like(arr),
            where=norms > 1e-9,
        )

    @staticmethod
    def _as_valid_vector(cached_vec: Any) -> np.ndarray | None:
        """캐시된 값이 1차원 벡터로 해석 가능하면 np.ndarray로, 아니면 None을 반환합니다."""
        try:
            vec_np = np.asarray(cached_vec, dtype="float32")
        except (TypeError, ValueError):
            return None
        if vec_np.ndim != 1 or vec_np.shape[0] == 0:
            return None
        return vec_np

    async def _reembed_single(self, text: str, expected_dim: int) -> np.ndarray:
        """차원 불일치/None 캐시 벡터에 대해 해당 텍스트를 재임베딩하여 복구합니다."""
        norm_text = " ".join(text.split())
        try:
            from core.resource_manager import get_resource_manager

            coordinator = get_resource_manager()
            async with coordinator.use_embedder(embedder=self.embedder):
                vecs = await asyncio.to_thread(
                    self.embedder.embed_documents, [norm_text]
                )
        except Exception as e:
            raise ValueError(f"[Chunker] 캐시 벡터 복구 재임베딩 실패: {e}") from e
        if not vecs:
            raise ValueError(
                f"[Chunker] 캐시 벡터 복구 재임베딩 결과가 비어 있습니다: {norm_text[:80]!r}"
            )
        vec_np = np.asarray(vecs[0], dtype="float32")
        if vec_np.ndim != 1 or vec_np.shape[0] != expected_dim:
            raise ValueError(
                f"[Chunker] 캐시 벡터 복구 재임베딩 후에도 차원 불일치 "
                f"(text: {norm_text[:80]!r}, expected dim: {expected_dim}, "
                f"got: {vec_np.shape})."
            )

        cache_key = (
            f"emb:{self.model_name}:{xxhash.xxh64(norm_text.encode()).hexdigest()}"
        )
        await asyncio.wait_for(
            self.cache_manager.set(
                cache_key,
                {"vector": vec_np.tolist(), "cache_version": "1.0"},
                persist_to_disk=True,
            ),
            timeout=30.0,
        )
        return vec_np

    @staticmethod
    def _resolve_expected_dim(
        all_results: list[np.ndarray | None],
        newly_embedded: list[tuple[int, str, np.ndarray]],
    ) -> int | None:
        """새로 임베딩된 벡터를 우선, 없으면 캐시 벡터로 표준 차원을 결정합니다."""
        for _, _, vec in newly_embedded:
            return int(vec.shape[0])
        for res_vec in all_results:
            if res_vec is not None:
                return int(res_vec.shape[0])
        return None

    def _buffer_combined_embeddings(self, indiv_embeddings: np.ndarray) -> np.ndarray:
        """문장별 임베딩에 buffer 기반 컨텍스트 윈도우를 적용한 combined 벡터를 계산합니다."""
        combined_embeddings = []
        for i in range(len(indiv_embeddings)):
            start = max(0, i - self.buffer_size)
            end = min(len(indiv_embeddings), i + self.buffer_size + 1)
            window_vectors = indiv_embeddings[start:end]
            combined_vec = np.mean(window_vectors, axis=0)
            norm = np.linalg.norm(combined_vec)
            if norm > 1e-9:
                combined_vec /= norm
            combined_embeddings.append(combined_vec)
        return np.array(combined_embeddings)

    def _compute_distances(self, combined_embeddings_arr: np.ndarray) -> list[float]:
        """인접 combined 벡터 간 유사도를 1-cos_sim 거리로 변환합니다 (벡터화)."""
        arr = np.asarray(combined_embeddings_arr, dtype=np.float64)
        if arr.shape[0] < 2:
            return []
        dots = np.einsum("ij,ij->i", arr[:-1], arr[1:])
        return (1.0 - dots).tolist()

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
            percentiles = cast(np.ndarray, np.percentile(dist_array, [25, 75]))
            q1, q3 = float(percentiles[0]), float(percentiles[1])
            iqr = q3 - q1
            threshold = float(
                np.mean(dist_array) + self.breakpoint_threshold_value * iqr
            )
        elif self.breakpoint_threshold_type == "gradient":
            threshold = float(
                np.percentile(np.gradient(dist_array), self.breakpoint_threshold_value)
            )
        elif self.breakpoint_threshold_type == "similarity_threshold":
            threshold = float(self.similarity_threshold)
        else:
            threshold = float(self.similarity_threshold)
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
            bp_set = set(breakpoints)
            for i, s in enumerate(sentences):
                current_len += len(s["text"])
                if current_len > (self.max_chunk_size * 0.9):
                    if (
                        (i not in bp_set)
                        and ((i + 1) not in bp_set)
                        and ((i + 2) not in bp_set)
                    ):
                        safety_bps.append(i + 1)
                    current_len = 0
            if safety_bps:
                breakpoints = sorted(set(breakpoints + safety_bps))

        return breakpoints

    def _optimize_chunk_sizes(self, chunks: list[dict]) -> list[dict]:
        """청크 크기를 검사해 병합하며, 벡터는 가중 평균으로 계산합니다.
        [수정] 섹션(Header)이 다르면 절대로 병합하지 않습니다.
        """
        if not chunks:
            return chunks

        optimized = []
        current_chunk = None

        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue

            merged_text = current_chunk["text"] + " " + chunk["text"]
            merged_len = len(merged_text)

            is_different_section = current_chunk.get("current_section") != chunk.get(
                "current_section"
            )

            is_at_hard_boundary = current_chunk.get("end") != chunk.get(
                "start"
            ) or current_chunk.get("is_hard_split", False)

            token_fits = True
            merged_tokens = None
            current_tokens = current_chunk.get("_token_count")
            chunk_tokens = chunk.get("_token_count")
            if (
                self.max_chunk_tokens is not None
                and current_tokens is not None
                and chunk_tokens is not None
            ):
                merged_tokens = int(current_tokens) + int(chunk_tokens) + 1
                token_fits = merged_tokens <= self.max_chunk_tokens

            should_merge = False

            if not is_different_section and not is_at_hard_boundary and token_fits:
                if len(current_chunk["text"]) < self.min_chunk_size:
                    should_merge = merged_len <= self.max_chunk_size

                if not should_merge and merged_len <= self.max_chunk_size:
                    sim = cosine_similarity(current_chunk["vector"], chunk["vector"])
                    if sim > (ChunkingConstants.SIMILARITY_MERGE_THRESHOLD / 100.0):
                        should_merge = True

            if should_merge:
                len_a = len(current_chunk["text"])
                len_b = len(chunk["text"])
                total_len = len_a + len_b

                if total_len > 0:
                    merged_vector = (
                        current_chunk["vector"] * len_a + chunk["vector"] * len_b
                    ) / total_len
                    norm = np.linalg.norm(merged_vector)
                    if norm > 0:
                        merged_vector /= norm
                else:
                    merged_vector = current_chunk["vector"]

                current_chunk["text"] = merged_text
                current_chunk["end"] = chunk["end"]
                current_chunk["vector"] = merged_vector
                current_chunk["is_hard_split"] = chunk.get("is_hard_split", False)
                if "_sentence_end" in chunk:
                    current_chunk["_sentence_end"] = chunk["_sentence_end"]
                if merged_tokens is not None:
                    current_chunk["_token_count"] = merged_tokens
            else:
                optimized.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            optimized.append(current_chunk)

        return optimized

    def _clean_section_title(self, raw_title: str) -> str:
        """섹션 제목에서 마크다운 기호 및 불필요한 본문 텍스트를 제거합니다."""
        clean_title = raw_title.lstrip("# ").strip()

        upper_title = clean_title.upper()
        for kw in STANDARD_SECTION_KEYWORDS:
            if upper_title.startswith(kw):
                return kw

        if len(clean_title) > 100:
            parts = re.split(r"[\n\r\.\:]", clean_title)
            if parts:
                clean_title = parts[0].strip()

        clean_title = re.sub(r"[#\*_]", "", clean_title)
        return clean_title[:150]

    def _prune_duplicates(
        self, chunks: list[dict], threshold: float = 0.98
    ) -> list[dict]:
        """임베딩 벡터 기반으로 유사도가 높은 중복 청크를 제거합니다."""
        if not chunks:
            return []

        pruned = [chunks[0]]
        for i in range(1, len(chunks)):
            current_vec = chunks[i]["vector"]
            is_dup = False

            for prev in pruned[-3:]:
                sim = cosine_similarity(current_vec, prev["vector"])
                if sim > threshold:
                    is_dup = True
                    break

            if not is_dup:
                pruned.append(chunks[i])
            else:
                logger.debug(f"[Chunker] 중복 청크 제거됨 (유사도 > {threshold})")

        return pruned

    async def _token_counts_for_texts(self, texts: list[str]) -> list[int] | None:
        """토큰 카운터가 설정된 경우에만 배치 토큰 수를 워커 스레드에서 계산한다."""
        if not texts or self._token_count_fn is None:
            return None
        return await asyncio.to_thread(self._token_count_fn, texts)

    def _make_chunk(
        self,
        sentences: list[dict],
        indiv_embeddings: np.ndarray,
        start: int,
        end: int,
        current_section: str,
        sentence_token_counts: list[int] | None = None,
    ) -> dict:
        """문장 인덱스 범위 [start, end)로 청크 dict를 조립한다."""
        merged_text = " ".join(s["text"] for s in sentences[start:end])
        chunk_vector = np.mean(indiv_embeddings[start:end], axis=0)
        chunk_norm = float(np.linalg.norm(chunk_vector))
        if chunk_norm > 1e-9:
            chunk_vector = chunk_vector / chunk_norm
        token_count = None
        if sentence_token_counts is not None:
            token_count = sum(sentence_token_counts[start:end]) + max(
                0, end - start - 1
            )
        return {
            "text": merged_text,
            "start": sentences[start]["start"],
            "end": sentences[end - 1]["end"],
            "vector": chunk_vector,
            "is_hard_split": sentences[end - 1].get("is_hard_split", False),
            "current_section": current_section,
            "_sentence_start": start,
            "_sentence_end": end,
            "_token_count": token_count,
        }

    def _split_chunk_by_tokens(
        self,
        chunk: dict,
        sentences: list[dict],
        indiv_embeddings: np.ndarray,
        sentence_token_counts: list[int],
        budget: int,
    ) -> list[dict]:
        """문장 경계를 유지한 채 토큰 예산 이하로 청크를 재분할한다."""
        start_idx = cast(int | None, chunk.get("_sentence_start"))
        end_idx = cast(int | None, chunk.get("_sentence_end"))
        if start_idx is None or end_idx is None or end_idx - start_idx <= 1:
            return [chunk]

        section = str(chunk.get("current_section", "일반 본문"))
        parts: list[dict] = []
        part_start = start_idx
        used = 0
        for idx in range(start_idx, end_idx):
            token_count = sentence_token_counts[idx] + (1 if idx > part_start else 0)
            if idx > part_start and used + token_count > budget:
                parts.append(
                    self._make_chunk(
                        sentences,
                        indiv_embeddings,
                        part_start,
                        idx,
                        section,
                        sentence_token_counts,
                    )
                )
                part_start = idx
                used = sentence_token_counts[idx]
            else:
                used += token_count
        if part_start < end_idx:
            parts.append(
                self._make_chunk(
                    sentences,
                    indiv_embeddings,
                    part_start,
                    end_idx,
                    section,
                    sentence_token_counts,
                )
            )
        return parts

    async def _enforce_chunk_token_limit(
        self,
        chunks: list[dict],
        sentences: list[dict],
        indiv_embeddings: np.ndarray,
        sentence_token_counts: list[int] | None,
    ) -> list[dict]:
        """최종 청크가 max_chunk_tokens를 넘지 않도록 문장 경계에서 재분할한다."""
        if not chunks or self.max_chunk_tokens is None or self._token_count_fn is None:
            return chunks

        limit = max(1, self.max_chunk_tokens)
        exact_counts = await self._token_counts_for_texts([c["text"] for c in chunks])
        if exact_counts is None:
            return chunks

        budget = max(1, limit - self.TOKEN_GROUP_MARGIN)
        out: list[dict] = []
        did_split = False
        for chunk, count in zip(chunks, exact_counts, strict=True):
            if count <= limit or sentence_token_counts is None:
                chunk["_token_count"] = count
                out.append(chunk)
                continue
            did_split = True
            out.extend(
                self._split_chunk_by_tokens(
                    chunk,
                    sentences,
                    indiv_embeddings,
                    sentence_token_counts,
                    budget,
                )
            )

        if did_split:
            final_counts = await self._token_counts_for_texts([c["text"] for c in out])
            if final_counts:
                for chunk, count in zip(out, final_counts, strict=True):
                    chunk["_token_count"] = count
            logger.info(
                "[Chunker] 토큰 예산 적용 | limit=%d | chunks=%d→%d | "
                "max_before=%d | max_after=%d",
                limit,
                len(chunks),
                len(out),
                max(exact_counts) if exact_counts else 0,
                max(final_counts) if final_counts else 0,
            )
        return out

    def _group_sentences_into_chunks(
        self,
        sentences: list[dict],
        header_indices: set[int],
        indiv_embeddings: np.ndarray,
        breakpoints: list[int],
        sentence_token_counts: list[int] | None = None,
    ) -> list[dict]:
        """분기점 기준으로 문장을 그룹화하고 헤더 컨텍스트(Section)를 추적해 청크를 조립합니다."""
        chunks = []
        start_idx = 0
        all_bps = breakpoints + [len(sentences)]

        current_header = "Front Matter"
        pending_header = ""
        is_first_header = True

        for i, bp in enumerate(all_bps):
            group_start = start_idx
            group_end = bp
            if group_start >= group_end:
                continue

            is_new_header_start = sentences[group_start].get("is_header", False)

            actual_start = group_start
            if i > 0 and not is_new_header_start:
                actual_start = max(0, group_start - self.chunk_overlap)

            for s_idx in range(start_idx, group_end):
                if s_idx not in header_indices:
                    continue
                s = sentences[s_idx]
                text_strip = s["text"].strip()
                new_h = self._clean_section_title(text_strip)

                incomplete_markers = ["OF", "AND", "WITH", "IN", "FOR", "THE", "A"]
                if (
                    any(new_h.upper().endswith(w) for w in incomplete_markers)
                    and len(new_h) < 100
                ):
                    pending_header = new_h + " "
                    continue

                if pending_header:
                    new_h = (pending_header + new_h).strip()
                    pending_header = ""

                if is_first_header and len(new_h) > 10:
                    current_header = f"TITLE: {new_h}"
                    is_first_header = False
                else:
                    current_header = new_h

            if sentence_token_counts is not None and self.max_chunk_tokens is not None:
                budget = max(1, self.max_chunk_tokens - self.TOKEN_GROUP_MARGIN)
                range_start = actual_start
                used = 0
                for idx in range(actual_start, group_end):
                    token_count = sentence_token_counts[idx] + (
                        1 if idx > range_start else 0
                    )
                    if idx > range_start and used + token_count > budget:
                        chunks.append(
                            self._make_chunk(
                                sentences,
                                indiv_embeddings,
                                range_start,
                                idx,
                                current_header,
                                sentence_token_counts,
                            )
                        )
                        range_start = idx
                        used = sentence_token_counts[idx]
                    else:
                        used += token_count
                if range_start < group_end:
                    chunks.append(
                        self._make_chunk(
                            sentences,
                            indiv_embeddings,
                            range_start,
                            group_end,
                            current_header,
                            sentence_token_counts,
                        )
                    )
            else:
                chunks.append(
                    self._make_chunk(
                        sentences,
                        indiv_embeddings,
                        actual_start,
                        group_end,
                        current_header,
                    )
                )
            start_idx = bp

        return chunks

    def _extract_metadata_for_chunk(
        self, chunk: dict[str, Any], doc_ranges: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        청크에 대한 메타데이터를 추출하고 병합합니다.
        """
        c_start = chunk["start"]
        c_end = chunk["end"]

        overlapping_pages = []
        seen_pages: set[Any] = set()
        merged_metadata: dict[str, Any] = {}

        idx = bisect.bisect_left(doc_ranges, c_start, key=lambda r: r["end"])
        while idx < len(doc_ranges):
            doc_range = doc_ranges[idx]
            if doc_range["start"] >= c_end:
                break
            if max(c_start, doc_range["start"]) < min(c_end, doc_range["end"]):
                metadata = cast(dict[str, Any], doc_range["metadata"])
                page = metadata.get("page")
                if page and page not in seen_pages:
                    seen_pages.add(page)
                    overlapping_pages.append(page)

                if not merged_metadata:
                    merged_metadata = metadata.copy()
            idx += 1

        if overlapping_pages:
            merged_metadata["pages"] = sorted(overlapping_pages)
            merged_metadata["page"] = overlapping_pages[0]
            merged_metadata["is_cross_page"] = len(overlapping_pages) > 1

        merged_metadata["current_section"] = chunk.get("current_section", "일반 본문")
        merged_metadata["start_index"] = c_start
        merged_metadata["end_index"] = c_end

        return merged_metadata

    def extract_sentence_units(self, text: str) -> tuple[list[dict], set[int]]:
        """문장 분할 + 짧은 문장 병합 (동기, 임베딩 없음)."""
        raw_sentences = self._split_sentences(text)
        if not raw_sentences:
            return [], set()
        return self._merge_short_sentences(raw_sentences)

    async def split_text(
        self, text: str, precomputed_embeddings: np.ndarray | None = None
    ) -> list[dict]:
        """텍스트를 의미론적으로 분할합니다."""
        if not text or not text.strip():
            return []

        sentences, header_indices = self.extract_sentence_units(text)
        if not sentences:
            return []

        sentence_token_counts = await self._token_counts_for_texts(
            [s["text"] for s in sentences]
        )

        if precomputed_embeddings is not None:
            if precomputed_embeddings.shape[0] != len(sentences):
                raise ValueError(
                    f"[Chunker] 사전 임베딩 행 수({precomputed_embeddings.shape[0]})가 "
                    f"문장 수({len(sentences)})와 일치하지 않습니다."
                )
            indiv_embeddings = np.asarray(precomputed_embeddings, dtype="float32")
            for s, v in zip(sentences, indiv_embeddings, strict=True):
                s["vector"] = v
            if len(sentences) <= self.buffer_size:
                return sentences
        elif len(sentences) <= self.buffer_size:
            if sentences:
                sentence_texts = [s["text"] for s in sentences]
                embeddings = await self._get_embeddings(sentence_texts, normalize=True)
                if embeddings.shape[0] != len(sentence_texts):
                    raise ValueError(
                        f"[Chunker] 임베딩 행 수({embeddings.shape[0]})가 "
                        f"문장 수({len(sentence_texts)})와 일치하지 않습니다."
                    )
                for s, v in zip(sentences, embeddings, strict=True):
                    s["vector"] = v
            return sentences
        else:
            indiv_embeddings = await self._get_embeddings(
                [s["text"] for s in sentences], normalize=True
            )
            if indiv_embeddings.shape[0] != len(sentences):
                raise ValueError(
                    f"[Chunker] 임베딩 행 수({indiv_embeddings.shape[0]})가 "
                    f"문장 수({len(sentences)})와 일치하지 않습니다."
                )
            for s, v in zip(sentences, indiv_embeddings, strict=True):
                s["vector"] = v

        combined_embeddings_arr = self._buffer_combined_embeddings(indiv_embeddings)
        distances = self._compute_distances(combined_embeddings_arr)
        breakpoints = self._find_breakpoints(distances, sentences=sentences)
        chunks = self._group_sentences_into_chunks(
            sentences,
            header_indices,
            indiv_embeddings,
            breakpoints,
            sentence_token_counts,
        )
        optimized_chunks = self._optimize_chunk_sizes(chunks)
        optimized_chunks = await self._enforce_chunk_token_limit(
            optimized_chunks,
            sentences,
            indiv_embeddings,
            sentence_token_counts,
        )
        return self._prune_duplicates(optimized_chunks)

    @staticmethod
    def _join_docs_text(docs: list["Document"]) -> tuple[str, list[dict]]:
        """문서들을 개행으로 결합하고 오프셋 매핑을 구축한다."""
        text_parts = []
        doc_ranges = []
        current_offset = 0
        _sep = "\n\n"

        for doc in docs:
            content = doc.page_content
            if not content:
                continue

            if current_offset > 0:
                text_parts.append(_sep)
                current_offset += len(_sep)

            start = current_offset
            text_parts.append(content)
            end = current_offset + len(content)

            doc_ranges.append({"start": start, "end": end, "metadata": doc.metadata})
            current_offset = end

        return "".join(text_parts), doc_ranges

    def prepare_overflow_texts(
        self, docs: list["Document"]
    ) -> tuple[str, list[dict], list[str]]:
        """오버플로우 문서 결합 + 문장 텍스트 추출 (임베딩 없음)."""
        full_text, doc_ranges = self._join_docs_text(docs)
        if not full_text.strip():
            return "", [], []
        sentences, _ = self.extract_sentence_units(full_text)
        return full_text, doc_ranges, [s["text"] for s in sentences]

    async def assemble_overflow_chunks(
        self,
        full_text: str,
        doc_ranges: list[dict],
        sentence_vectors: np.ndarray,
    ) -> tuple[list["Document"], list[np.ndarray]]:
        """사전 임베딩된 문장 벡터로 의미분할을 완성한다."""
        chunk_dicts = await self.split_text(
            full_text, precomputed_embeddings=sentence_vectors
        )
        return await self.build_output_documents(chunk_dicts, doc_ranges)

    async def build_output_documents(
        self, chunk_dicts: list[dict], doc_ranges: list[dict]
    ) -> tuple[list["Document"], list[np.ndarray]]:
        """청크 딕셔너리에 메타데이터를 역매핑해 Document·벡터 쌍을 조립한다."""

        def _extract_metadata_batch(
            chunk_dicts: list[dict], doc_ranges: list[dict]
        ) -> list[dict]:
            return [
                self._extract_metadata_for_chunk(chunk, doc_ranges)
                for chunk in chunk_dicts
            ]

        final_docs: list[Document] = []
        final_vectors: list[np.ndarray] = []

        results = await asyncio.to_thread(
            _extract_metadata_batch, chunk_dicts, doc_ranges
        )

        for chunk, merged_metadata in zip(chunk_dicts, results, strict=False):
            final_docs.append(
                Document(page_content=chunk["text"], metadata=merged_metadata)
            )
            final_vectors.append(chunk["vector"])

        return final_docs, final_vectors

    async def split_documents(
        self, docs: list["Document"]
    ) -> tuple[list["Document"], list[np.ndarray]]:
        """
        LangChain Document 객체 리스트를 받아 의미론적 분할을 수행합니다.
        """
        if not docs:
            logger.warning("split_documents: 입력 문서 리스트가 비어있습니다.")
            return [], []

        full_text, doc_ranges = self._join_docs_text(docs)

        if not full_text.strip():
            logger.warning("split_documents: 병합된 텍스트가 비어있습니다.")
            return [], []

        chunk_dicts = await self.split_text(full_text)

        final_docs, final_vectors = await self.build_output_documents(
            chunk_dicts, doc_ranges
        )

        logger.info(
            f"의미론적 문서 분할 완료: {len(docs)}개 원본 문서 -> {len(final_docs)}개 청크 생성 (벡터 포함)"
        )
        return final_docs, final_vectors
