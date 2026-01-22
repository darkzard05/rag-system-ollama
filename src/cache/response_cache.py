"""
응답 캐싱 - Task 13
RAG 시스템에 최적화된 캐싱
"""

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import asyncio

from services.optimization.caching_optimizer import CacheManager, CacheStatistics, get_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class QueryCacheKey:
    """쿼리 캐시 키"""
    query: str
    document_ids: List[str] = None
    
    def to_hash(self) -> str:
        """해시값 계산"""
        data = {
            "query": self.query,
            "docs": sorted(self.document_ids) if self.document_ids else []
        }
        key = json.dumps(data, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class ResponseCacheEntry:
    """응답 캐시 항목"""
    query: str
    response: str
    metadata: Dict[str, Any]
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "query": self.query,
            "response": self.response,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class ResponseCache:
    """
    LLM 응답 캐싱
    
    특징:
    - 질의 결과 캐싱
    - TTL 기반 만료
    - 의미적 유사 쿼리 검색
    - 캐시 통계
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        default_ttl_hours: int = 3
    ):
        self.cache_manager = cache_manager or get_cache_manager()
        self.default_ttl = default_ttl_hours * 3600  # 초 단위
        self.access_log = []
    
    async def get(
        self,
        query: str,
        use_semantic: bool = True
    ) -> Optional[ResponseCacheEntry]:
        """
        응답 조회
        
        Args:
            query: 쿼리
            use_semantic: 의미적 유사성 사용 여부
        
        Returns:
            캐시 항목 또는 None
        """
        try:
            cache_key = self._generate_key(query)
            
            # 메모리 캐시 확인
            result = await self.cache_manager.get(cache_key, use_semantic=False)
            if result:
                logger.debug(f"[ResponseCache] 정확 일치 캐시 히트: {query[:50]}")
                return result
            
            # 세맨틱 캐시 확인
            if use_semantic:
                result = await self.cache_manager.get(
                    query,
                    use_semantic=True
                )
                if result:
                    logger.debug(f"[ResponseCache] 의미 유사 캐시 히트: {query[:50]}")
                    return result
            
            logger.debug(f"[ResponseCache] 캐시 미스: {query[:50]}")
            return None
        
        except Exception as e:
            logger.error(f"[ResponseCache] 조회 오류: {e}")
            return None
    
    async def set(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None
    ) -> None:
        """
        응답 저장
        
        Args:
            query: 쿼리
            response: 응답
            metadata: 메타데이터
            ttl_hours: TTL (시간 단위)
        """
        try:
            cache_key = self._generate_key(query)
            ttl_seconds = (ttl_hours or 3) * 3600
            
            entry = ResponseCacheEntry(
                query=query,
                response=response,
                metadata=metadata or {},
                created_at=datetime.now().timestamp()
            )
            
            # 메모리 + 세맨틱 캐시에 저장
            await self.cache_manager.set(
                cache_key,
                entry,
                ttl_seconds=ttl_seconds,
                use_semantic=True
            )
            
            # 의미 기반 캐시에도 저장
            await self.cache_manager.set(
                query,
                entry,
                ttl_seconds=ttl_seconds,
                use_semantic=True
            )
            
            logger.debug(f"[ResponseCache] 응답 저장: {query[:50]}")
            self.access_log.append({
                "timestamp": datetime.now(),
                "query": query[:100],
                "type": "set",
                "response_length": len(response)
            })
        
        except Exception as e:
            logger.error(f"[ResponseCache] 저장 오류: {e}")
    
    async def delete(self, query: str) -> None:
        """응답 삭제"""
        try:
            cache_key = self._generate_key(query)
            await self.cache_manager.delete(cache_key)
            await self.cache_manager.delete(query)
            logger.debug(f"[ResponseCache] 응답 삭제: {query[:50]}")
        except Exception as e:
            logger.error(f"[ResponseCache] 삭제 오류: {e}")
    
    async def clear(self) -> None:
        """전체 응답 캐시 삭제"""
        await self.cache_manager.clear()
        logger.info("[ResponseCache] 모든 응답 캐시 삭제")
    
    def get_stats(self) -> Dict[str, CacheStatistics]:
        """통계 조회"""
        return self.cache_manager.get_stats()
    
    def get_combined_stats(self) -> CacheStatistics:
        """통합 통계"""
        return self.cache_manager.get_combined_stats()
    
    def _generate_key(self, query: str) -> str:
        """캐시 키 생성"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]


class QueryCache:
    """
    검색 쿼리 결과 캐싱
    
    특징:
    - 문서 검색 결과 캐싱
    - 쿼리별 문서 목록 저장
    - 캐시 무효화 전략
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        default_ttl_hours: int = 24
    ):
        self.cache_manager = cache_manager or get_cache_manager()
        self.default_ttl = default_ttl_hours * 3600
        self.invalidation_callbacks: List[Callable] = []
    
    async def get(
        self,
        query: str,
        top_k: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        검색 결과 조회
        
        Args:
            query: 검색 쿼리
            top_k: 상위 K개 문서
        
        Returns:
            문서 목록 또는 None
        """
        try:
            cache_key = f"query:{self._generate_key(query)}:top{top_k}"
            result = await self.cache_manager.get(cache_key, use_semantic=False)
            
            if result:
                logger.debug(f"[QueryCache] 검색 결과 캐시 히트: {query[:50]}")
                return result
            
            logger.debug(f"[QueryCache] 검색 결과 캐시 미스: {query[:50]}")
            return None
        
        except Exception as e:
            logger.error(f"[QueryCache] 조회 오류: {e}")
            return None
    
    async def set(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        ttl_hours: Optional[int] = None
    ) -> None:
        """
        검색 결과 저장
        
        Args:
            query: 검색 쿼리
            documents: 검색된 문서 목록
            top_k: 상위 K개
            ttl_hours: TTL (시간 단위)
        """
        try:
            cache_key = f"query:{self._generate_key(query)}:top{top_k}"
            ttl_seconds = (ttl_hours or 24) * 3600
            
            await self.cache_manager.set(
                cache_key,
                documents,
                ttl_seconds=ttl_seconds,
                use_semantic=False
            )
            
            logger.debug(
                f"[QueryCache] 검색 결과 저장: {query[:50]} "
                f"({len(documents)} 문서)"
            )
        
        except Exception as e:
            logger.error(f"[QueryCache] 저장 오류: {e}")
    
    async def invalidate(self, query: Optional[str] = None) -> None:
        """
        캐시 무효화
        
        Args:
            query: 특정 쿼리만 무효화 (None이면 전체)
        """
        try:
            if query:
                cache_key = f"query:{self._generate_key(query)}:*"
                await self.cache_manager.delete(cache_key)
                logger.debug(f"[QueryCache] 무효화: {query}")
            else:
                await self.cache_manager.clear()
                logger.info("[QueryCache] 모든 쿼리 캐시 무효화")
            
            # 콜백 실행
            for callback in self.invalidation_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(query)
                    else:
                        callback(query)
                except Exception as e:
                    logger.error(f"[QueryCache] 무효화 콜백 오류: {e}")
        
        except Exception as e:
            logger.error(f"[QueryCache] 무효화 오류: {e}")
    
    def register_invalidation_callback(
        self,
        callback: Callable
    ) -> None:
        """무효화 콜백 등록"""
        self.invalidation_callbacks.append(callback)
    
    def _generate_key(self, query: str) -> str:
        """캐시 키 생성"""
        return hashlib.sha256(query.encode()).hexdigest()[:12]


class DocumentCache:
    """
    문서 캐싱
    
    특징:
    - 문서별 임베딩 캐싱
    - 문서 내용 캐싱
    - 청킹 결과 캐싱
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        default_ttl_hours: int = 7 * 24  # 7일
    ):
        self.cache_manager = cache_manager or get_cache_manager()
        self.default_ttl = default_ttl_hours * 3600
    
    async def get_document(
        self,
        doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """문서 조회"""
        try:
            cache_key = f"doc:{doc_id}"
            result = await self.cache_manager.get(cache_key, use_semantic=False)
            
            if result:
                logger.debug(f"[DocumentCache] 문서 캐시 히트: {doc_id}")
                return result
            
            return None
        
        except Exception as e:
            logger.error(f"[DocumentCache] 조회 오류: {e}")
            return None
    
    async def set_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
        ttl_hours: Optional[int] = None
    ) -> None:
        """문서 저장"""
        try:
            cache_key = f"doc:{doc_id}"
            ttl_seconds = (ttl_hours or 7 * 24) * 3600
            
            await self.cache_manager.set(
                cache_key,
                document,
                ttl_seconds=ttl_seconds,
                use_semantic=False
            )
            
            logger.debug(f"[DocumentCache] 문서 저장: {doc_id}")
        
        except Exception as e:
            logger.error(f"[DocumentCache] 저장 오류: {e}")
    
    async def get_chunks(
        self,
        doc_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """청킹 결과 조회"""
        try:
            cache_key = f"chunks:{doc_id}"
            result = await self.cache_manager.get(cache_key, use_semantic=False)
            
            if result:
                logger.debug(f"[DocumentCache] 청크 캐시 히트: {doc_id}")
                return result
            
            return None
        
        except Exception as e:
            logger.error(f"[DocumentCache] 청크 조회 오류: {e}")
            return None
    
    async def set_chunks(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]],
        ttl_hours: Optional[int] = None
    ) -> None:
        """청킹 결과 저장"""
        try:
            cache_key = f"chunks:{doc_id}"
            ttl_seconds = (ttl_hours or 7 * 24) * 3600
            
            await self.cache_manager.set(
                cache_key,
                chunks,
                ttl_seconds=ttl_seconds,
                use_semantic=False
            )
            
            logger.debug(f"[DocumentCache] 청크 저장: {doc_id} ({len(chunks)} 청크)")
        
        except Exception as e:
            logger.error(f"[DocumentCache] 청크 저장 오류: {e}")
    
    async def invalidate_document(self, doc_id: str) -> None:
        """문서 캐시 무효화"""
        try:
            await self.cache_manager.delete(f"doc:{doc_id}")
            await self.cache_manager.delete(f"chunks:{doc_id}")
            logger.debug(f"[DocumentCache] 문서 무효화: {doc_id}")
        except Exception as e:
            logger.error(f"[DocumentCache] 무효화 오류: {e}")


class CacheWarmup:
    """
    캐시 사전 로딩
    
    특징:
    - 자주 사용되는 쿼리 미리 캐싱
    - 벤치마크 쿼리 세트 저장
    - 캐시 초기화
    """
    
    def __init__(
        self,
        response_cache: ResponseCache,
        query_cache: QueryCache
    ):
        self.response_cache = response_cache
        self.query_cache = query_cache
        self.warmup_queries: List[Dict[str, Any]] = []
    
    def add_warmup_query(
        self,
        query: str,
        response: str,
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """워밍업 쿼리 추가"""
        self.warmup_queries.append({
            "query": query,
            "response": response,
            "documents": documents or []
        })
    
    async def warmup(self) -> int:
        """캐시 사전 로딩 실행"""
        count = 0
        
        try:
            for item in self.warmup_queries:
                query = item["query"]
                response = item["response"]
                documents = item.get("documents", [])
                
                # 응답 캐싱
                await self.response_cache.set(query, response)
                
                # 쿼리 캐싱
                if documents:
                    await self.query_cache.set(query, documents)
                
                count += 1
                logger.debug(f"[CacheWarmup] 캐싱 완료: {query[:50]}")
            
            logger.info(f"[CacheWarmup] {count}개 쿼리 캐싱 완료")
            return count
        
        except Exception as e:
            logger.error(f"[CacheWarmup] 워밍업 오류: {e}")
            return count
    
    def clear(self) -> None:
        """워밍업 쿼리 목록 초기화"""
        self.warmup_queries.clear()


# 전역 캐시 인스턴스
_response_cache: Optional[ResponseCache] = None
_query_cache: Optional[QueryCache] = None
_document_cache: Optional[DocumentCache] = None


def get_response_cache() -> ResponseCache:
    """응답 캐시 인스턴스"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def get_query_cache() -> QueryCache:
    """쿼리 캐시 인스턴스"""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


def get_document_cache() -> DocumentCache:
    """문서 캐시 인스턴스"""
    global _document_cache
    if _document_cache is None:
        _document_cache = DocumentCache()
    return _document_cache


def reset_caches() -> None:
    """모든 캐시 리셋"""
    global _response_cache, _query_cache, _document_cache
    _response_cache = None
    _query_cache = None
    _document_cache = None
