"""
인덱스 최적화 모듈 - 메모리 효율성, 검색 성능 향상.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from threading import RLock
import numpy as np

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """벡터 압축 방법."""
    NONE = "none"
    QUANTIZATION_INT8 = "quantization_int8"
    QUANTIZATION_INT4 = "quantization_int4"
    PRODUCT_QUANTIZATION = "product_quantization"


class IndexOptimizationStrategy(Enum):
    """인덱스 최적화 전략."""
    STANDARD = "standard"
    MEMORY_EFFICIENT = "memory_efficient"
    SPEED_OPTIMIZED = "speed_optimized"
    BALANCED = "balanced"


@dataclass
class VectorQuantizationConfig:
    """벡터 양자화 설정."""
    compression_method: CompressionMethod = CompressionMethod.QUANTIZATION_INT8
    target_bits: int = 8  # 4 또는 8
    preserve_norm: bool = True
    enable_scaling: bool = True


@dataclass
class IndexOptimizationConfig:
    """인덱스 최적화 설정."""
    strategy: IndexOptimizationStrategy = IndexOptimizationStrategy.BALANCED
    quantization_config: VectorQuantizationConfig = field(default_factory=VectorQuantizationConfig)
    enable_lru_cache: bool = True
    max_cache_size: int = 1000
    enable_doc_pruning: bool = False
    min_doc_similarity: float = 0.95
    enable_metadata_indexing: bool = True


@dataclass
class IndexStats:
    """인덱스 통계."""
    total_documents: int = 0
    total_vectors: int = 0
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 0.0
    avg_search_time_ms: float = 0.0
    num_unique_metadata_fields: int = 0
    pruned_documents: int = 0


class VectorQuantizer:
    """벡터 양자화 엔진."""
    
    def __init__(self, config: VectorQuantizationConfig):
        self.config = config
        self._lock = RLock()
    
    def quantize_vectors(self, vectors: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        벡터 양자화.
        
        Args:
            vectors: 원본 벡터 리스트 (shape: [n, d])
        
        Returns:
            (양자화된 벡터, 메타데이터)
        """
        if self.config.compression_method == CompressionMethod.NONE:
            return vectors, {"method": "none"}
        
        if self.config.compression_method == CompressionMethod.QUANTIZATION_INT8:
            return self._quantize_int8(vectors)
        elif self.config.compression_method == CompressionMethod.QUANTIZATION_INT4:
            return self._quantize_int4(vectors)
        elif self.config.compression_method == CompressionMethod.PRODUCT_QUANTIZATION:
            return self._product_quantization(vectors)
    
    def _quantize_int8(self, vectors: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """INT8 양자화."""
        quantized = []
        scales = []
        offsets = []
        
        for vec in vectors:
            if self.config.preserve_norm:
                norm = np.linalg.norm(vec)
            
            # Min-max 정규화
            vec_min = np.min(vec)
            vec_max = np.max(vec)
            scale = (vec_max - vec_min) / 255.0 if (vec_max - vec_min) > 0 else 1.0
            offset = vec_min
            
            # 양자화
            quantized_vec = np.round((vec - offset) / scale).astype(np.int8)
            quantized.append(quantized_vec)
            
            scales.append(scale)
            offsets.append(offset)
            
            if self.config.preserve_norm:
                scales[-1] *= norm
        
        metadata = {
            "method": "quantization_int8",
            "scales": scales,
            "offsets": offsets,
            "compression_ratio": 4.0,  # float32 → int8
        }
        
        return quantized, metadata
    
    def _quantize_int4(self, vectors: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """INT4 양자화 (더 높은 압축)."""
        quantized = []
        scales = []
        offsets = []
        
        for vec in vectors:
            # Min-max 정규화
            vec_min = np.min(vec)
            vec_max = np.max(vec)
            scale = (vec_max - vec_min) / 15.0 if (vec_max - vec_min) > 0 else 1.0
            offset = vec_min
            
            # 양자화 (4-bit는 uint8로 표현, 실제로는 두 개의 4-bit 값)
            quantized_vec = np.round((vec - offset) / scale).astype(np.uint8)
            quantized.append(quantized_vec)
            
            scales.append(scale)
            offsets.append(offset)
        
        metadata = {
            "method": "quantization_int4",
            "scales": scales,
            "offsets": offsets,
            "compression_ratio": 8.0,  # float32 → int4 (packed as uint8)
        }
        
        return quantized, metadata
    
    def _product_quantization(self, vectors: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """Product Quantization (PQ)."""
        # 간단한 PQ 구현
        quantized = []
        codebooks = []
        assignments = []
        
        num_subspaces = 4
        
        for vec in vectors:
            # 부분공간별로 나누기
            subspace_size = len(vec) // num_subspaces
            pq_codes = []
            
            for i in range(num_subspaces):
                start_idx = i * subspace_size
                end_idx = (i + 1) * subspace_size if i < num_subspaces - 1 else len(vec)
                subvec = vec[start_idx:end_idx]
                
                # K-means 클러스터링으로 가장 가까운 코드북 찾기
                code = np.argmin(np.abs(subvec - np.mean(subvec)))
                pq_codes.append(int(code % 256))  # 256 클러스터
            
            quantized.append(np.array(pq_codes, dtype=np.uint8))
            assignments.append(pq_codes)
        
        metadata = {
            "method": "product_quantization",
            "num_subspaces": num_subspaces,
            "compression_ratio": 2.0,
        }
        
        return quantized, metadata
    
    def dequantize_vectors(
        self,
        quantized_vectors: List[np.ndarray],
        metadata: Dict[str, Any],
    ) -> List[np.ndarray]:
        """양자화된 벡터 복원."""
        method = metadata.get("method", "none")
        
        if method == "none":
            return quantized_vectors
        elif method == "quantization_int8":
            return self._dequantize_int8(quantized_vectors, metadata)
        elif method == "quantization_int4":
            return self._dequantize_int4(quantized_vectors, metadata)
        
        return quantized_vectors
    
    def _dequantize_int8(
        self,
        quantized_vectors: List[np.ndarray],
        metadata: Dict,
    ) -> List[np.ndarray]:
        """INT8 복원."""
        scales = metadata.get("scales", [])
        offsets = metadata.get("offsets", [])
        
        dequantized = []
        for q_vec, scale, offset in zip(quantized_vectors, scales, offsets):
            vec = (q_vec.astype(np.float32) * scale) + offset
            dequantized.append(vec)
        
        return dequantized
    
    def _dequantize_int4(
        self,
        quantized_vectors: List[np.ndarray],
        metadata: Dict,
    ) -> List[np.ndarray]:
        """INT4 복원."""
        scales = metadata.get("scales", [])
        offsets = metadata.get("offsets", [])
        
        dequantized = []
        for q_vec, scale, offset in zip(quantized_vectors, scales, offsets):
            vec = (q_vec.astype(np.float32) * scale) + offset
            dequantized.append(vec)
        
        return dequantized


class DocumentPruner:
    """중복/유사 문서 제거기."""
    
    def __init__(self, min_similarity: float = 0.95):
        self.min_similarity = min_similarity
        self._lock = RLock()
    
    def prune_similar_documents(
        self,
        documents: List[Document],
        vectors: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[Document], List[int]]:
        """
        유사도 기반 문서 제거.
        
        Returns:
            (제거 후 문서 리스트, 제거된 인덱스)
        """
        if not documents or not vectors:
            return documents, []
        
        kept_indices = []
        removed_indices = []
        
        for i, (doc_i, vec_i) in enumerate(zip(documents, vectors)):
            is_duplicate = False
            
            for j in kept_indices:
                doc_j = documents[j]
                vec_j = vectors[j]
                
                # 코사인 유사도
                similarity = self._cosine_similarity(vec_i, vec_j)
                
                if similarity >= self.min_similarity:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_indices.append(i)
            else:
                removed_indices.append(i)
        
        pruned_docs = [documents[i] for i in kept_indices]
        return pruned_docs, removed_indices
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """코사인 유사도 계산."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)


class MetadataIndexer:
    """메타데이터 인덱싱 엔진."""
    
    def __init__(self):
        self.indexes: Dict[str, Dict[Any, Set[int]]] = {}
        self._lock = RLock()
    
    def build_indexes(self, documents: List[Document]):
        """메타데이터 인덱스 구축."""
        with self._lock:
            self.indexes.clear()
            
            # 모든 메타데이터 필드 인덱싱
            for doc_idx, doc in enumerate(documents):
                for key, value in doc.metadata.items():
                    if key not in self.indexes:
                        self.indexes[key] = {}
                    
                    if value not in self.indexes[key]:
                        self.indexes[key][value] = set()
                    
                    self.indexes[key][value].add(doc_idx)
    
    def search_by_metadata(
        self,
        field: str,
        value: Any,
    ) -> Set[int]:
        """메타데이터로 문서 검색."""
        with self._lock:
            if field not in self.indexes:
                return set()
            
            if value not in self.indexes[field]:
                return set()
            
            return self.indexes[field][value].copy()
    
    def get_stats(self) -> Dict[str, int]:
        """메타데이터 인덱스 통계."""
        with self._lock:
            return {
                field: len(values)
                for field, values in self.indexes.items()
            }


class LRUVectorCache:
    """LRU 벡터 캐시."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order: List[str] = []
        self._lock = RLock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """캐시에서 벡터 조회."""
        with self._lock:
            if key not in self.cache:
                return None
            
            # LRU 순서 업데이트
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return self.cache[key].copy()
    
    def put(self, key: str, vector: np.ndarray):
        """캐시에 벡터 저장."""
        with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
            
            # 용량 초과시 LRU 제거
            if len(self.cache) >= self.max_size and key not in self.cache:
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = vector.copy()
            self.access_order.append(key)
    
    def clear(self):
        """캐시 초기화."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """캐시 통계."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            }


class IndexOptimizer:
    """인덱스 최적화 통합 관리자."""
    
    def __init__(self, config: IndexOptimizationConfig):
        self.config = config
        self.quantizer = VectorQuantizer(config.quantization_config)
        self.pruner = DocumentPruner(config.min_doc_similarity)
        self.metadata_indexer = MetadataIndexer()
        self.vector_cache = LRUVectorCache(config.max_cache_size) if config.enable_lru_cache else None
        
        self._lock = RLock()
        self.stats = IndexStats()
    
    def optimize_index(
        self,
        documents: List[Document],
        vectors: List[np.ndarray],
    ) -> Tuple[List[Document], List[np.ndarray], IndexStats]:
        """
        인덱스 최적화.
        
        Returns:
            (최적화된 문서, 최적화된 벡터, 통계)
        """
        self.stats = IndexStats(
            total_documents=len(documents),
            total_vectors=len(vectors),
        )
        
        # 1. 문서 프루닝
        if self.config.enable_doc_pruning:
            documents, pruned_indices = self.pruner.prune_similar_documents(
                documents, vectors
            )
            vectors = [v for i, v in enumerate(vectors) if i not in pruned_indices]
            self.stats.pruned_documents = len(pruned_indices)
        
        # 2. 벡터 양자화
        original_size = sum(v.nbytes for v in vectors)
        vectors, quant_metadata = self.quantizer.quantize_vectors(vectors)
        compressed_size = sum(v.nbytes for v in vectors)
        
        self.stats.original_size_mb = original_size / (1024 * 1024)
        self.stats.compressed_size_mb = compressed_size / (1024 * 1024)
        self.stats.compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else 1.0
        )
        
        # 3. 메타데이터 인덱싱
        if self.config.enable_metadata_indexing:
            self.metadata_indexer.build_indexes(documents)
            self.stats.num_unique_metadata_fields = len(self.metadata_indexer.indexes)
        
        return documents, vectors, self.stats
    
    def search_with_metadata(
        self,
        field: str,
        value: Any,
        documents: List[Document],
    ) -> List[Tuple[int, Document]]:
        """메타데이터 기반 검색."""
        doc_indices = self.metadata_indexer.search_by_metadata(field, value)
        return [
            (idx, documents[idx])
            for idx in sorted(doc_indices)
        ]
    
    def get_stats(self) -> IndexStats:
        """현재 인덱스 통계."""
        return self.stats


# 전역 인스턴스
_optimizer_instance: Optional[IndexOptimizer] = None


def get_index_optimizer(config: Optional[IndexOptimizationConfig] = None) -> IndexOptimizer:
    """전역 인덱스 최적화 인스턴스 조회."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = IndexOptimizer(config or IndexOptimizationConfig())
    return _optimizer_instance


def reset_index_optimizer():
    """인덱스 최적화 인스턴스 리셋 (테스트용)."""
    global _optimizer_instance
    _optimizer_instance = None
