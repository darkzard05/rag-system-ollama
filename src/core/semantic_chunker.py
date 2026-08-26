"""
임베딩 기반 의미론적 텍스트 분할기 — 오케스트레이션 모듈.

이 모듈은 문서를 문장 단위로 우선 분할한 후, 인접 문장 간의 임베딩 유사도를
계산하여 유사도가 낮은 지점을 경계로 선택합니다. 이를 통해 의미론적으로
일관성 있는 청크를 생성합니다.

[R2-10] 958줄 모노리스를 관심사별 믹스인 모듈로 분리했습니다.
본 모듈은 클래스 뼈대 + ``__init__`` + ``split_text``/``split_documents``
오케스트레이션만 담당합니다.

- ``semantic_chunker_sentences.py``: 문장 분할/리플로우/짧은 문장 병합
- ``semantic_chunker_embeddings.py``: 임베딩 생성/캐시/거리 계산
- ``semantic_chunker_breakpoints.py``: 브레이크포인트 탐색/헤더 패턴
- ``semantic_chunker_merge.py``: 청크 병합/중복제거/그룹화/섹션 제목
- ``semantic_chunker_metadata.py``: 청크→문서 메타데이터 역매핑

공개 API(``EmbeddingBasedSemanticChunker.split_documents``·``_split_sentences``·
``__init__`` 시그니처)는 기존과 동일하게 유지되어 외부 import 경로가 그대로
동작합니다.
"""

import asyncio
import concurrent.futures
import logging
import re
from pathlib import Path
from typing import cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import MODEL_CACHE_DIR
from core.semantic_chunker_breakpoints import SemanticChunkerBreakpointsMixin
from core.semantic_chunker_embeddings import SemanticChunkerEmbeddingsMixin
from core.semantic_chunker_merge import SemanticChunkerMergeMixin
from core.semantic_chunker_metadata import SemanticChunkerMetadataMixin
from core.semantic_chunker_sentences import SemanticChunkerSentencesMixin
from services.optimization.caching_optimizer import CacheManager

logger = logging.getLogger(__name__)


class EmbeddingBasedSemanticChunker(
    SemanticChunkerSentencesMixin,
    SemanticChunkerEmbeddingsMixin,
    SemanticChunkerBreakpointsMixin,
    SemanticChunkerMergeMixin,
    SemanticChunkerMetadataMixin,
):
    """
    임베딩 기반 의미론적 텍스트 분할기.

    문장 단위로 분할 후, 임베딩 유사도 기반으로 의미 경계를 탐지하여
    일관성 있는 청크를 생성합니다.
    """

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

        # [최적화] 청킹 임베딩 전용 캐시 관리자 설정.
        # 변경 전에는 메모리 전용(enable_disk_cache=False) + persist_to_disk=False
        # 였어서, 재시작/재수집 시 동일 청크라도 항상 Ollama 왕복이 발생했다.
        # 이제 영구 디스크 캐시(L3)를 켜 동일 콘텐츠 해시 키로 재수집 시 Ollama
        # 라운드트립을 건너뛴다. 메모리 레이어(L1)도 함께 두어 동일 프로세스 내
        # 반복 조회는 디스크 I/O 없이 처리한다. 디스크 경로는 전역 MODEL_CACHE_DIR
        # 하위의 embedding_cache 로 격리해 기타 응답 캐시와 혼재되지 않게 한다.
        self.cache_manager = cache_manager or CacheManager(
            enable_memory_cache=True,
            enable_semantic_cache=False,
            enable_disk_cache=True,
            disk_cache_dir=str(Path(MODEL_CACHE_DIR) / "embedding_cache"),
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

    async def split_text(self, text: str) -> list[dict]:
        """
        텍스트를 의미론적으로 분할합니다 (Buffer-based context window 적용).
        [고도화] 마크다운 헤더 뿐만 아니라 대문자 섹션명도 감지하여 구조를 파악합니다.
        """
        if not text or not text.strip():
            return []

        # 1. 문장 분할 (오프셋 포함)
        raw_sentences = self._split_sentences(text)
        if not raw_sentences:
            return []

        # [최적화] 너무 짧은 문장 병합 (오프셋 유지) + 헤더 위치 캐싱
        sentences, header_indices = self._merge_short_sentences(raw_sentences)

        if len(sentences) <= self.buffer_size:
            # 벡터가 없으므로 계산 필요
            if sentences:
                sentence_texts = [s["text"] for s in sentences]
                # [R2-02] 단위노름 배선 — 병합/중복제거의 dot product가 진짜 코사인 유사도
                embeddings = await self._get_embeddings(sentence_texts, normalize=True)
                if embeddings.shape[0] != len(sentence_texts):
                    raise ValueError(
                        f"[Chunker] 임베딩 행 수({embeddings.shape[0]})가 "
                        f"문장 수({len(sentence_texts)})와 일치하지 않습니다."
                    )
                for s, v in zip(sentences, embeddings, strict=True):
                    s["vector"] = v
            return sentences

        # 2. 개별 문장 임베딩 생성 (캐싱 활용)
        # [R2-02] 단위노름 배선 — 거리·병합·중복제거 일관성 확보
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

        # 3-4. Buffer 기반 Combined Embeddings + 인접 거리 계산
        combined_embeddings_arr = self._buffer_combined_embeddings(indiv_embeddings)
        distances = self._compute_distances(combined_embeddings_arr)

        # 5. 분기점 탐색 (헤더 인식 로직 포함됨)
        breakpoints = self._find_breakpoints(distances, sentences=sentences)

        # 6. 그룹화 및 헤더 컨텍스트(Section) 추적
        chunks = self._group_sentences_into_chunks(
            sentences, header_indices, indiv_embeddings, breakpoints
        )

        # 7. 크기 최적화 및 중복 제거
        optimized_chunks = self._optimize_chunk_sizes(chunks)
        return self._prune_duplicates(optimized_chunks)

    async def split_documents(
        self, docs: list["Document"]
    ) -> tuple[list["Document"], list[np.ndarray]]:
        """
        LangChain Document 객체 리스트를 받아 의미론적 분할을 수행합니다.
        문서들을 통합하여 문맥을 유지하되, 오프셋 매핑을 통해
        각 청크의 원본 메타데이터(페이지 번호 등)를 정확히 보존합니다.
        각 청크에 대해 이미 계산된 벡터를 함께 반환하여 재계산을 방지합니다.
        """

        if not docs:
            logger.warning("split_documents: 입력 문서 리스트가 비어있습니다.")
            return [], []

        # 1. 문서 통합 및 오프셋 매핑 구축 (join()을 사용하여 효율성 개선)
        text_parts = []
        doc_ranges = []
        current_offset = 0

        for doc in docs:
            content = doc.page_content
            if not content:
                continue

            # 문서 사이 공백 추가 (첫 문서 제외)
            if current_offset > 0:
                text_parts.append(" ")
                current_offset += 1

            start = current_offset
            text_parts.append(content)
            end = current_offset + len(content)

            doc_ranges.append({"start": start, "end": end, "metadata": doc.metadata})
            current_offset = end

        full_text = "".join(text_parts)

        if not full_text.strip():
            logger.warning("split_documents: 병합된 텍스트가 비어있습니다.")
            return [], []

        # 2. 청킹 수행 (오프셋 및 벡터 정보 포함)
        chunk_dicts = await self.split_text(full_text)

        # 3. 메타데이터 역매핑 및 문서 객체 생성 (다중 페이지 지원)
        final_docs = []
        final_vectors = []

        def _extract_metadata_batch(
            chunk_dicts: list[dict], doc_ranges: list[dict]
        ) -> list[dict]:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return list(
                    executor.map(
                        lambda chunk: self._extract_metadata_for_chunk(
                            chunk, doc_ranges
                        ),
                        chunk_dicts,
                    )
                )

        results = await asyncio.to_thread(
            _extract_metadata_batch, chunk_dicts, doc_ranges
        )

        for chunk, merged_metadata in zip(chunk_dicts, results, strict=False):
            final_docs.append(
                Document(page_content=chunk["text"], metadata=merged_metadata)
            )
            final_vectors.append(chunk["vector"])

        logger.info(
            f"의미론적 문서 분할 완료: {len(docs)}개 원본 문서 -> {len(final_docs)}개 청크 생성 (벡터 포함)"
        )
        return final_docs, final_vectors
