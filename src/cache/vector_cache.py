"""
벡터 저장소 캐싱을 담당하는 모듈.
"""

import json
import logging
import os
import shutil
import uuid
from typing import Any

import orjson
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import (
    RETRIEVER_CONFIG,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
    VECTOR_STORE_CONFIG,
)
from common.text_utils import bm25_tokenizer
from common.utils import fast_hash
from security.cache_security import (
    CacheIntegrityError,
    CacheTrustError,
)
from services.optimization.caching_optimizer import (
    ObjectCache,
    SyncCacheBridge,
)

logger = logging.getLogger(__name__)

# [R3a-03] 캐시 서명 스키마 버전. 상향 시 저장된 VERSION.json과 로드 시
# 재검증(:216-240)이 새 버전을 인식해 구버전 캐시는 자연 퇴거(재구축)됩니다.
CACHE_SCHEMA_VERSION: int = 3

# [R3a-03] semantic chunker의 임베딩 정규화 배선 (T8/R2-02에서 단위노름으로
# 일원화 — src/core/semantic_chunker.py:711,724의 normalize=True). 이 배선이
# 바뀌면(예: normalize=False) 청크 벡터가 달라지므로 서명에 반영해 스테일
# 캐시 히트를 방지합니다. 배선 변경 시 이 상수도 함께 갱신해야 합니다.
SEMANTIC_CHUNKER_NORMALIZE: bool = True

# [R3a-03] retriever_factory(src/core/retriever_factory.py:106-112)가
# index_params에서 소비하는 키의 기본값 — 미설정(빈 dict)과 명시 기본값이
# 동일 서명을 생성하도록 서명 직렬화 시 병합됩니다.
_INDEX_PARAM_DEFAULTS: dict[str, Any] = {
    "use_l2_norm": True,
    "hnsw_m": 32,
    "quantization_threshold": 5000,
    "nprobe": 16,
}


def _normalize_index_params(index_params: dict) -> dict[str, Any]:
    """index_params를 타입 안정하게 정규화해 서명 해시 안정성을 보장합니다.

    소비처가 실제 적용하는 기본값을 병합하고 모든 값을 bool/float/str 계열로
    일원화합니다. orjson 직렬화 시 int(5000)와 float(5000.0)가 다른 바이트로
    인코딩되어 동일 설정이 다른 서명을 만들 수 있으므로, 수치 키는 float로,
    플래그는 bool로 고정합니다.
    """
    merged = {**_INDEX_PARAM_DEFAULTS, **index_params}
    normalized: dict[str, Any] = {}
    for key, value in merged.items():
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (int, float)):
            normalized[key] = float(value)
        elif isinstance(value, (list, tuple)):
            normalized[key] = [str(v) for v in value]
        else:
            normalized[key] = str(value)
    return normalized


def _serialize_docs(docs: list[Document]) -> list[dict]:
    """Pydantic의 무거운 dict() 대신 직접 필요한 필드만 추출"""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]


def _deserialize_docs(doc_dicts: list[dict]) -> list[Document]:
    """dict 리스트를 Document 객체 리스트로 변환"""
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in doc_dicts
    ]


# [R3a-06] Pickle-free 정책 — 저장·로드 양쪽에서 pickle(index.pkl)을 생성/사용하지
# 않는다. langchain의 `FAISS.save_local`은 `index.faiss`와 함께 pickle `index.pkl`
# (docstore + index_to_docstore_id 2-튜플)을 디스크에 작성하지만, 로드 경로
# (load())는 index.pkl을 읽지 않고 index.faiss + JSON으로 재구성한다. 따라서
# index.pkl은 데드 아티팩트 + 잠재 보안 표면이므로 저장 포맷은 `faiss.write_index`
# (바이너리 인덱스) + JSON 직렬화(docstore 매핑)로 단일화한다.
def _write_pickle_free_index(vector_store: Any, faiss_index_dir: str) -> None:
    """FAISS 인덱스와 docstore 매핑을 pickle 없이 디스크에 저장합니다.

    - 바이너리 인덱스는 `faiss.write_index`로 `index.faiss`에 기록합니다
      (save_local 내부 구현과 동일한 직렬화 — 인덱스 타입/메트릭/검색 파라미터
      보존).
    - docstore 매핑(index position -> doc_id)은 JSON(`index_to_docstore_id.json`)
      으로 직렬화해 로드 시 동일 매핑으로 docstore를 재구성할 수 있게 합니다.
      문서 본문은 `doc_splits.json`이 이미 JSON으로 담당합니다.
    """
    import faiss

    os.makedirs(faiss_index_dir, exist_ok=True)
    index_path = os.path.join(faiss_index_dir, "index.faiss")
    faiss.write_index(vector_store.index, index_path)

    mapping_path = os.path.join(faiss_index_dir, "index_to_docstore_id.json")
    mapping = {
        str(position): doc_id
        for position, doc_id in sorted(vector_store.index_to_docstore_id.items())
    }
    with open(mapping_path, "wb") as f:
        f.write(orjson.dumps(mapping))


def _reconstruct_doc_ids(faiss_index_dir: str, doc_count: int) -> list[str]:
    """저장된 JSON docstore 매핑으로 doc_id 목록을 재구성합니다.

    Pickle-free 정책: `index_to_docstore_id.json`(R3a-06 저장 포맷)이 있으면
    doc_id 일관성을 보존하고, 없으면(legacy 캐시) 기존처럼 uuid4로 재생성합니다.
    """
    mapping_path = os.path.join(faiss_index_dir, "index_to_docstore_id.json")
    if not os.path.exists(mapping_path):
        return [str(uuid.uuid4()) for _ in range(doc_count)]
    with open(mapping_path, "rb") as f:
        raw_mapping = orjson.loads(f.read())
    index_to_docstore_id = {
        int(position): doc_id for position, doc_id in raw_mapping.items()
    }
    return [index_to_docstore_id[pos] for pos in sorted(index_to_docstore_id)]


def _build_cache_payloads(
    doc_splits: list[Document], bm25_docs: list[Document]
) -> dict[str, bytes]:
    """문서 캐시 payload를 한 번만 생성해 재사용합니다."""
    serialized_splits = _serialize_docs(doc_splits)
    serialized_bm25 = _serialize_docs(bm25_docs)
    return {
        "doc_splits_payload": orjson.dumps(serialized_splits),
        "bm25_payload": orjson.dumps(serialized_bm25),
    }


def _build_config_signature(embedding_model_name: str) -> str:
    """청킹/파싱 설정 기반의 안정적인 캐시 키 서명을 생성합니다.

    임베딩 모델명, 청크 크기/중첩, 분할 구분자, 의미론적 분할 사용 여부, 인덱스
    전략(index_params), 청커 정규화 배선이 바뀌면 서로 다른 캐시 디렉터리를
    사용하도록 하여 오래된 설정으로 생성된 청크가 재사용되지 않도록 보장합니다.
    (설정 변경 시 일회성 캐시 재구축)
    """
    splitter = TEXT_SPLITTER_CONFIG
    semantic = SEMANTIC_CHUNKER_CONFIG
    payload = orjson.dumps(
        {
            "embedding_model": str(embedding_model_name),
            "chunk_size": splitter.get("chunk_size"),
            "chunk_overlap": splitter.get("chunk_overlap"),
            "separators": list(splitter.get("separators") or []),
            "semantic_chunker_enabled": semantic.get("enabled", False),
            # [R3a-03] 인덱스 전략 — use_l2_norm/hnsw_m/quantization_threshold/nprobe
            # (retriever_factory 소비 키)가 바뀌면 스테일 인덱스 캐시를 방지해야 함.
            "index_params": _normalize_index_params(
                VECTOR_STORE_CONFIG.get("index_params", {})
            ),
            # [R3a-03] semantic chunker 정규화 배선(T8) — 정규화 전후 벡터가
            # 달라도 캐시 히트되지 않도록 서명에 반영.
            "semantic_chunker_normalize": SEMANTIC_CHUNKER_NORMALIZE,
        },
        option=orjson.OPT_SORT_KEYS,
    )
    return fast_hash(payload.decode())


class VectorStoreCache:
    """
    벡터 저장소와 관련 컴포넌트를 디스크에 캐싱하고 로드합니다.
    Pickle-free 정책: 저장(save)과 로드(load) 양쪽 모두 pickle(index.pkl)을
    생성/사용하지 않습니다 — 인덱스는 faiss.write_index, docstore 매핑은 JSON으로
    직렬화해 보안성을 강화합니다 (R3a-06).
    """

    def __init__(
        self,
        file_path: str,
        embedding_model_name: str,
        cache_dir: str = VECTOR_STORE_CACHE_DIR,
        file_hash: str | None = None,
    ):
        from core.document_processor import compute_file_hash

        self.file_hash = file_hash or compute_file_hash(file_path)
        cache_key = _build_config_signature(embedding_model_name)
        self.cache_dir = os.path.join(cache_dir, f"{self.file_hash}_{cache_key}")
        self.doc_splits_path = os.path.join(self.cache_dir, "doc_splits.json")
        self.faiss_index_path = os.path.join(self.cache_dir, "faiss_index")
        self.bm25_retriever_path = os.path.join(self.cache_dir, "bm25_docs.json")

        # Backward compatibility: legacy cache key using [:10] truncation
        legacy_key = embedding_model_name[:10]
        self._legacy_cache_dir: str | None = None
        if legacy_key != cache_key:
            self._legacy_cache_dir = os.path.join(
                cache_dir, f"{self.file_hash}_{legacy_key}"
            )

        from security.cache_security import get_security_manager

        self.security_manager = get_security_manager()
        self._cache_invalid = False

        # [R4/R9] 통합 객체 캐시 브릿지 — 파싱된 인메모리 객체(doc_splits/
        # vector_store/bm25_retriever)를 SyncCacheBridge 로 감싼 ObjectCache 에
        # 보관합니다. load/save 는 평범한 def(비동기 아님)이므로 await 할 수 없고,
        # 호출부(pipeline_builder.py)도 동기 컨텍스트라 이벤트 루프 충돌을 피하려면
        # SyncCacheBridge 가 필요합니다. 디스크 직렬화(pickle-free)는 그대로 유지.
        self._object_cache: SyncCacheBridge = SyncCacheBridge(
            ObjectCache[tuple[list[Document] | None, Any | None, Any | None]](
                max_size=256, ttl_seconds=0.0
            )
        )

    def _try_load_legacy(self) -> bool:
        """Check if legacy cache dir (pre-fast_hash) exists and re-point paths."""
        if not self._legacy_cache_dir:
            return False
        legacy_paths = (
            os.path.join(self._legacy_cache_dir, "doc_splits.json"),
            os.path.join(self._legacy_cache_dir, "faiss_index"),
            os.path.join(self._legacy_cache_dir, "bm25_docs.json"),
        )
        if not all(os.path.exists(p) for p in legacy_paths):
            return False
        self.cache_dir = self._legacy_cache_dir
        self.doc_splits_path = legacy_paths[0]
        self.faiss_index_path = legacy_paths[1]
        self.bm25_retriever_path = legacy_paths[2]
        return True

    def load(
        self,
        embedder: Embeddings,
        resource_manager: Any | None = None,
    ) -> tuple[list[Document] | None, Any | None, Any | None]:
        # [R4/R9] 통합 객체 캐시: 디스크에서 파싱된 인메모리 객체를 먼저 조회해
        # 중복 파싱(특히 FAISS 인덱스 로드)을 방지합니다. 키는 캐시 디렉터리 경로
        # (legacy 재지정 포함)로, 동일 파일/설정이면 동일 객체를 반환합니다.
        object_cache_key = self.cache_dir
        cached = self._object_cache.get_sync(object_cache_key)
        if cached is not None:
            logger.debug(f"[Cache] Unified object-cache hit: {object_cache_key}")
            return cached

        if not all(
            os.path.exists(p)
            for p in [
                self.doc_splits_path,
                self.faiss_index_path,
                self.bm25_retriever_path,
            ]
        ):
            # Backward compat: try legacy cache dir (pre-fast_hash)
            if self._legacy_cache_dir and self._try_load_legacy():
                logger.info(
                    f"[Cache] Loaded from legacy cache dir: {self._legacy_cache_dir}"
                )
            else:
                return None, None, None

        try:
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_community.retrievers import BM25Retriever
            from langchain_community.vectorstores import FAISS
            from langchain_community.vectorstores.utils import DistanceStrategy

            paths_to_verify = [
                (self.doc_splits_path, "문서 데이터"),
                (self.faiss_index_path, "FAISS 인덱스"),
                (self.bm25_retriever_path, "BM25 리트리버"),
            ]

            for path, desc in paths_to_verify:
                try:
                    if not self.security_manager.verify_cache_trust(path):
                        raise CacheTrustError(f"불신 경로: {path}")
                    if os.path.isfile(path):
                        if os.path.exists(path + ".meta"):
                            self.security_manager.verify_cache_integrity(path)
                        else:
                            # [SEC-FIX] .meta 누락 = 무결성 검증 불가 → fail-closed
                            # (기존 "생략" 분기는 조작된 아티팩트 무검증 로드 허용).
                            raise CacheIntegrityError(
                                f".meta 누락으로 무결성 검증 불가: {path}"
                            )
                    elif os.path.isdir(path):
                        for f in os.listdir(path):
                            if f.endswith(".meta"):
                                continue  # 메타데이터 사이드카는 검증 대상 아님
                            f_path = os.path.join(path, f)
                            if os.path.exists(f_path + ".meta"):
                                self.security_manager.verify_cache_integrity(f_path)
                            else:
                                raise CacheIntegrityError(
                                    f".meta 누락으로 무결성 검증 불가: {f_path}"
                                )
                except (CacheTrustError, CacheIntegrityError) as e:
                    logger.critical(
                        f"[Security] 캐시 무결성 위반 감지 ({desc}: {type(e).__name__}). "
                        "캐시를 삭제하지 않고 재구축을 유도합니다."
                    )
                    self._cache_invalid = True
                    return None, None, None

            # 1. 문서 조각 로드
            with open(self.doc_splits_path, "rb") as file_handle:
                doc_dicts = orjson.loads(file_handle.read())
            doc_splits = _deserialize_docs(doc_dicts)

            # 2. FAISS 인덱스 수동 로드 (Pickle/index.pkl 무시)
            index_file = os.path.join(self.faiss_index_path, "index.faiss")
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")

            # FAISS 버전 호환성 검사
            version_path = os.path.join(self.faiss_index_path, "VERSION.json")
            if os.path.exists(version_path):
                try:
                    with open(version_path) as _f:
                        manifest = json.load(_f)
                    stored_faiss_version = manifest.get("faiss_version", "unknown")
                    if stored_faiss_version != faiss.__version__:
                        logger.warning(
                            f"[CACHE] FAISS version mismatch: cached={stored_faiss_version}, "
                            f"runtime={faiss.__version__}. 캐시를 삭제하지 않고 재구축합니다."
                        )
                        self._cache_invalid = True
                        return None, None, None
                    # [R3a-03] 스키마 버전 재검증 — 구버전(schema < 3) 캐시는
                    # 서명 변경을 반영하지 못하므로 자연 퇴거(재구축)를 유도한다.
                    stored_schema_version = manifest.get("schema_version", 1)
                    if stored_schema_version < CACHE_SCHEMA_VERSION:
                        logger.warning(
                            f"[CACHE] 스키마 버전 불일치: cached={stored_schema_version}, "
                            f"runtime={CACHE_SCHEMA_VERSION}. 캐시를 삭제하지 않고 재구축합니다."
                        )
                        self._cache_invalid = True
                        return None, None, None
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.warning(
                        f"[CACHE] VERSION.json read failed ({e}). "
                        "캐시를 삭제하지 않고 재구축합니다."
                    )
                    self._cache_invalid = True
                    return None, None, None
            else:
                # Legacy cache without version info — assume compatible
                logger.warning(
                    "[CACHE] No VERSION.json found (legacy cache). Proceeding with compatibility mode."
                )

            index = faiss.read_index(index_file)

            # [R3a-06] index.pkl 대신 JSON으로 직렬화된 docstore 매핑으로 doc_id를
            # 재구성합니다 (legacy 캐시는 기존처럼 uuid4 재생성으로 폴백).
            doc_ids = _reconstruct_doc_ids(self.faiss_index_path, len(doc_splits))
            new_docstore_docs = dict(zip(doc_ids, doc_splits, strict=False))
            docstore = InMemoryDocstore(new_docstore_docs)
            index_to_docstore_id = dict(enumerate(doc_ids))

            vector_store = FAISS(
                embedding_function=embedder,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )

            # 3. BM25 로드
            with open(self.bm25_retriever_path, "rb") as file_handle:
                bm25_doc_dicts = orjson.loads(file_handle.read())
            bm25_docs = _deserialize_docs(bm25_doc_dicts)

            bm25_retriever = BM25Retriever.from_documents(
                bm25_docs, preprocess_func=bm25_tokenizer
            )
            bm25_retriever.k = RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 5)

            logger.info(f"RAG 캐시 안전 로드 완료 (Pickle-free): '{self.cache_dir}'")
            result = (doc_splits, vector_store, bm25_retriever)
            # [R4/R9] 파싱 완료된 객체를 통합 캐시에 보관 (동기 브릿지 경유).
            self._object_cache.set_sync(object_cache_key, result)
            return result

        except Exception as e:
            logger.warning(
                f"캐시 로드 중 예외 발생: {e}. 캐시를 삭제하지 않고 재구축을 유도합니다."
            )
            self._cache_invalid = True
            return None, None, None

    def save(
        self,
        doc_splits: list[Document],
        vector_store: Any,
        bm25_retriever: Any,
    ) -> None:
        if os.path.exists(self.cache_dir) and not self._cache_invalid:
            logger.info(f"[Cache] 캐시가 이미 존재함: {self.cache_dir}")
            return

        staging_dir = f"{self.cache_dir}.tmp.{uuid.uuid4().hex[:8]}"
        stg_doc_splits_path = os.path.join(staging_dir, "doc_splits.json")
        stg_faiss_index_path = os.path.join(staging_dir, "faiss_index")
        stg_bm25_retriever_path = os.path.join(staging_dir, "bm25_docs.json")

        try:
            os.makedirs(staging_dir, exist_ok=True)
            self.security_manager.enforce_directory_permissions(staging_dir)

            bm25_docs = getattr(bm25_retriever, "docs", None)
            if bm25_docs is None:
                bm25_docs = doc_splits

            payloads = _build_cache_payloads(doc_splits, bm25_docs)
            with open(stg_doc_splits_path, "wb") as f:
                f.write(payloads["doc_splits_payload"])
            self.security_manager.enforce_file_permissions(stg_doc_splits_path)

            doc_meta = self.security_manager.create_metadata_for_file(
                stg_doc_splits_path, description="Document splits cache (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_doc_splits_path + ".meta", doc_meta
            )

            # [R3a-06] save_local 대신 faiss.write_index + JSON docstore 매핑으로
            # 저장해 데드 pickle 아티팩트(index.pkl)를 생성하지 않습니다.
            _write_pickle_free_index(vector_store, stg_faiss_index_path)
            self.security_manager.enforce_directory_permissions(stg_faiss_index_path)

            # FAISS 버전 메타데이터 저장 (버전 불일치 시 자동 캐시 재생성)
            import sys as _sys
            import time as _time

            import faiss as _faiss

            version_manifest = {
                "faiss_version": _faiss.__version__,
                "schema_version": CACHE_SCHEMA_VERSION,
                "python_version": _sys.version,
                "created_at": _time.time(),
            }
            version_path = os.path.join(stg_faiss_index_path, "VERSION.json")
            with open(version_path, "w") as _f:
                json.dump(version_manifest, _f, indent=2)

            # load()는 .meta가 존재하는 파일만 무결성 검증하므로 모든 아티팩트에 생성.
            for artifact in os.listdir(stg_faiss_index_path):
                artifact_path = os.path.join(stg_faiss_index_path, artifact)
                if os.path.isfile(artifact_path):
                    meta = self.security_manager.create_metadata_for_file(
                        artifact_path,
                        description=f"FAISS cache artifact ({artifact})",
                    )
                    self.security_manager.save_cache_metadata(
                        artifact_path + ".meta", meta
                    )

            # BM25 저장
            with open(stg_bm25_retriever_path, "wb") as f:
                f.write(payloads["bm25_payload"])
            self.security_manager.enforce_file_permissions(stg_bm25_retriever_path)
            bm25_meta = self.security_manager.create_metadata_for_file(
                stg_bm25_retriever_path, description="BM25 retriever cache (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_bm25_retriever_path + ".meta", bm25_meta
            )

            # 스테이징 디렉토리를 최종 위치로 이동
            if self._cache_invalid and os.path.exists(self.cache_dir):
                # 비파괴 재구축: 무결성 위반으로 판정된 캐시만 교체 대상.
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            try:
                os.rename(staging_dir, self.cache_dir)
            except FileExistsError:
                # 다른 프로세스가 먼저 저장한 경우: 스테이징만 정리하고 종료.
                logger.info(
                    "[Cache] 동시 저장 감지 (다른 프로세스가 먼저 저장). 스테이징 제거."
                )
                if os.path.exists(staging_dir):
                    shutil.rmtree(staging_dir)
                return
            self._cache_invalid = False
            # [R4/R9] 디스크에 새로 쓴 상태와 인메모리 객체 캐시를 일치시킵니다.
            # 파싱 객체는 새 disk 상태에서 load 시 재구성되므로 여기선 무효화.
            self._object_cache.delete_sync(self.cache_dir)
            logger.info(f"[Cache] 벡터 캐시 저장 완료: {self.cache_dir}")

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            # 실패 시 잠재적 스테일 인메모리 객체 캐시도 정리.
            self._object_cache.delete_sync(self.cache_dir)
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
