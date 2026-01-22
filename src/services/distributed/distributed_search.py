"""
Task 19-1: Distributed Search Execution Module
분산 검색 실행 - 다중 노드에서 병렬 검색 수행
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import time
import threading
from threading import RLock, Thread
import uuid
import heapq


class SearchStatus(Enum):
    """검색 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeSearchStatus(Enum):
    """노드별 검색 상태"""
    IDLE = "idle"
    SEARCHING = "searching"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SearchResult:
    """개별 검색 결과"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """점수 기반 비교 (내림차순)"""
        return self.score > other.score


@dataclass
class SearchQuery:
    """검색 쿼리 정의"""
    query_id: str
    query_text: str
    embedding: List[float]
    top_k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class NodeSearchTask:
    """노드별 검색 작업"""
    task_id: str
    node_id: str
    query: SearchQuery
    status: NodeSearchStatus = NodeSearchStatus.IDLE
    results: List[SearchResult] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class NodeSearchEngine:
    """개별 노드의 검색 엔진"""
    
    def __init__(self, node_id: str, db_size: int = 1000):
        """
        Args:
            node_id: 노드 ID
            db_size: 해당 노드의 문서 데이터베이스 크기
        """
        self.node_id = node_id
        self.db_size = db_size
        self._documents = self._init_documents()
        self._lock = RLock()
    
    def _init_documents(self) -> List[Dict[str, Any]]:
        """검색을 위한 모의 문서 초기화"""
        docs = []
        for i in range(self.db_size):
            docs.append({
                'id': f"{self.node_id}_doc_{i}",
                'content': f"Document {i} on node {self.node_id}",
                'embedding': [float(j % 10) / 10 for j in range(384)],  # Mock embedding
                'metadata': {
                    'source': f'source_{i % 5}',
                    'date': f'2026-01-{(i % 28) + 1:02d}',
                    'category': f'cat_{i % 3}'
                }
            })
        return docs
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        쿼리로 문서 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 리스트
        """
        with self._lock:
            results = []
            
            for doc in self._documents:
                # 모의 유사도 계산
                score = self._calculate_similarity(query.embedding, doc['embedding'])
                
                # 필터 적용
                if not self._apply_filters(doc, query.filters):
                    continue
                
                result = SearchResult(
                    doc_id=doc['id'],
                    content=doc['content'],
                    score=score,
                    metadata=doc['metadata'],
                    node_id=self.node_id
                )
                results.append(result)
            
            # 점수 기반 정렬 및 top-k 선택
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:query.top_k]
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not embedding1 or not embedding2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(x ** 2 for x in embedding1) ** 0.5
        norm2 = sum(x ** 2 for x in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _apply_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """필터 적용"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key not in doc['metadata']:
                return False
            if doc['metadata'][key] != value:
                return False
        
        return True
    
    def search_with_filter(self, query: SearchQuery, extra_filter: Optional[Dict] = None) -> List[SearchResult]:
        """필터가 추가된 검색"""
        if extra_filter:
            query.filters.update(extra_filter)
        return self.search(query)


class DistributedSearchExecutor:
    """분산 검색 실행기 - 다중 노드에서 병렬 검색"""
    
    def __init__(self, num_nodes: int = 3, docs_per_node: int = 100):
        """
        Args:
            num_nodes: 노드 개수
            docs_per_node: 노드당 문서 수
        """
        self.num_nodes = num_nodes
        self.docs_per_node = docs_per_node
        
        # 각 노드의 검색 엔진
        self._nodes = {
            f"node_{i}": NodeSearchEngine(f"node_{i}", docs_per_node)
            for i in range(num_nodes)
        }
        
        # 진행 중인 검색 작업
        self._active_searches: Dict[str, SearchQuery] = {}
        self._node_tasks: Dict[str, List[NodeSearchTask]] = {node_id: [] for node_id in self._nodes}
        self._node_status: Dict[str, NodeSearchStatus] = {
            node_id: NodeSearchStatus.IDLE for node_id in self._nodes
        }
        self._search_results: Dict[str, List[SearchResult]] = {}
        
        self._lock = RLock()
    
    def submit_query(self, query_text: str, embedding: List[float], top_k: int = 10) -> str:
        """
        검색 쿼리 제출
        
        Args:
            query_text: 검색 텍스트
            embedding: 쿼리 임베딩
            top_k: 상위 k개 결과
            
        Returns:
            쿼리 ID
        """
        query_id = str(uuid.uuid4())
        query = SearchQuery(
            query_id=query_id,
            query_text=query_text,
            embedding=embedding,
            top_k=top_k
        )
        
        with self._lock:
            self._active_searches[query_id] = query
            self._search_results[query_id] = []
        
        return query_id
    
    def execute_search_on_node(self, node_id: str, query: SearchQuery) -> SearchResult:
        """
        특정 노드에서 검색 실행
        
        Args:
            node_id: 노드 ID
            query: 검색 쿼리
            
        Returns:
            검색 결과 리스트
        """
        if node_id not in self._nodes:
            raise ValueError(f"Unknown node: {node_id}")
        
        task_id = str(uuid.uuid4())
        task = NodeSearchTask(
            task_id=task_id,
            node_id=node_id,
            query=query,
            status=NodeSearchStatus.SEARCHING,
            start_time=time.time()
        )
        
        try:
            with self._lock:
                self._node_status[node_id] = NodeSearchStatus.SEARCHING
            
            # 실제 검색 실행
            results = self._nodes[node_id].search(query)
            
            task.status = NodeSearchStatus.COMPLETED
            task.results = results
            task.end_time = time.time()
            task.execution_time = task.end_time - task.start_time
            
        except Exception as e:
            task.status = NodeSearchStatus.ERROR
            task.error_message = str(e)
            task.end_time = time.time()
        
        finally:
            with self._lock:
                self._node_tasks[node_id].append(task)
                self._node_status[node_id] = task.status
        
        return task
    
    def execute_distributed_search(self, query_id: str) -> List[SearchResult]:
        """
        모든 노드에서 분산 검색 실행
        
        Args:
            query_id: 쿼리 ID
            
        Returns:
            전체 검색 결과 (노드별 결과 병합)
        """
        with self._lock:
            if query_id not in self._active_searches:
                raise ValueError(f"Unknown query: {query_id}")
            
            query = self._active_searches[query_id]
        
        # 각 노드에서 병렬로 검색 실행
        all_results = []
        for node_id in self._nodes:
            task = self.execute_search_on_node(node_id, query)
            all_results.extend(task.results)
        
        # 결과 저장
        with self._lock:
            self._search_results[query_id] = all_results
        
        return all_results
    
    def execute_parallel_search(self, query_id: str, num_threads: Optional[int] = None) -> List[SearchResult]:
        """
        쓰레드 기반 병렬 검색
        
        Args:
            query_id: 쿼리 ID
            num_threads: 병렬 쓰레드 수 (기본값: 노드 수)
            
        Returns:
            병합된 검색 결과
        """
        if num_threads is None:
            num_threads = len(self._nodes)
        
        with self._lock:
            if query_id not in self._active_searches:
                raise ValueError(f"Unknown query: {query_id}")
            
            query = self._active_searches[query_id]
        
        all_results = []
        results_lock = RLock()
        
        def search_worker(node_id: str):
            """검색 워커"""
            task = self.execute_search_on_node(node_id, query)
            with results_lock:
                all_results.extend(task.results)
        
        # 쓰레드 풀 생성 및 실행
        threads = []
        for node_id in self._nodes:
            thread = Thread(target=search_worker, args=(node_id,), daemon=True)
            threads.append(thread)
            thread.start()
        
        # 모든 쓰레드 대기
        for thread in threads:
            thread.join(timeout=query.timeout)
        
        with self._lock:
            self._search_results[query_id] = all_results
        
        return all_results
    
    def get_search_results(self, query_id: str) -> List[SearchResult]:
        """검색 결과 조회"""
        with self._lock:
            return self._search_results.get(query_id, [])
    
    def get_node_status(self, node_id: str) -> NodeSearchStatus:
        """노드 상태 조회"""
        with self._lock:
            return self._node_status.get(node_id, NodeSearchStatus.IDLE)
    
    def get_node_tasks(self, node_id: str) -> List[NodeSearchTask]:
        """노드의 검색 작업 내역 조회"""
        with self._lock:
            return list(self._node_tasks.get(node_id, []))
    
    def cancel_search(self, query_id: str) -> bool:
        """검색 취소"""
        with self._lock:
            if query_id in self._active_searches:
                del self._active_searches[query_id]
                return True
        return False
    
    def get_search_statistics(self, query_id: str) -> Dict[str, Any]:
        """검색 통계 조회"""
        with self._lock:
            if query_id not in self._search_results:
                return {}
            
            results = self._search_results[query_id]
            
            if not results:
                return {
                    'total_results': 0,
                    'nodes_searched': 0,
                    'avg_score': 0.0,
                    'score_range': (0.0, 0.0)
                }
            
            scores = [r.score for r in results]
            results_by_node = {}
            for result in results:
                if result.node_id not in results_by_node:
                    results_by_node[result.node_id] = 0
                results_by_node[result.node_id] += 1
            
            return {
                'total_results': len(results),
                'nodes_searched': len(results_by_node),
                'avg_score': sum(scores) / len(scores),
                'score_range': (min(scores), max(scores)),
                'results_per_node': results_by_node
            }
    
    def filter_results_by_metadata(self, query_id: str, filter_key: str, filter_value: str) -> List[SearchResult]:
        """메타데이터 기반 결과 필터링"""
        with self._lock:
            results = self._search_results.get(query_id, [])
        
        filtered = [
            r for r in results
            if r.metadata.get(filter_key) == filter_value
        ]
        return filtered


class SearchQueryBuilder:
    """검색 쿼리 빌더"""
    
    def __init__(self):
        self.query_text = ""
        self.embedding = []
        self.top_k = 10
        self.filters = {}
    
    def set_query(self, text: str) -> 'SearchQueryBuilder':
        """쿼리 텍스트 설정"""
        self.query_text = text
        return self
    
    def set_embedding(self, embedding: List[float]) -> 'SearchQueryBuilder':
        """임베딩 설정"""
        self.embedding = embedding
        return self
    
    def set_top_k(self, k: int) -> 'SearchQueryBuilder':
        """top-k 설정"""
        self.top_k = k
        return self
    
    def add_filter(self, key: str, value: str) -> 'SearchQueryBuilder':
        """필터 추가"""
        self.filters[key] = value
        return self
    
    def build(self) -> SearchQuery:
        """쿼리 빌드"""
        return SearchQuery(
            query_id=str(uuid.uuid4()),
            query_text=self.query_text,
            embedding=self.embedding,
            top_k=self.top_k,
            filters=self.filters
        )
