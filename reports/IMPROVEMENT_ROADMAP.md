# 📋 RAG System 코드 개선 작업 로드맵

## 🎯 개선 작업 우선순위 (Pickle 보안 제외)

### 🔴 **P1 - 즉시 적용 (1-2일)**

#### 1. 타임아웃 처리 강화
- **파일**: `src/graph_builder.py` (L72-88)
- **문제**: 검색 및 LLM 호출에 타임아웃 없음
- **영향**: 무한 대기 가능성
- **난이도**: ⭐ 낮음
- **작업 내용**:
  ```python
  # 현재
  return await retriever.ainvoke(q)
  
  # 개선
  return await asyncio.wait_for(
      retriever.ainvoke(q),
      timeout=30.0
  )
  ```

#### 2. 상수 정의 (매직 숫자 제거)
- **파일**: `src/` 전체
- **문제**: 64, 600, 650, 500 등 하드코딩된 값
- **영향**: 유지보수 어려움, 오류 가능성 높음
- **난이도**: ⭐ 낮음
- **작업 내용**:
  - `src/constants.py` 생성
  - `UIConstants`, `PerformanceConstants`, `ChunkingConstants` 클래스 정의
  - 모든 매직 숫자 대체

#### 3. 기본 로깅 설정
- **파일**: `src/logging_config.py` (새 파일)
- **문제**: 로그 수준이 일관되지 않음
- **난이도**: ⭐ 낮음
- **작업 내용**:
  - 중앙 집중식 로깅 설정
  - 로그 파일 자동 로테이션
  - 구조화된 로그 포맷

---

### 🟡 **P2 - 중기 적용 (3-5일)**

#### 4. 고급 오류 처리
- **파일**: `src/rag_core.py` (L322-328)
- **문제**: 모호한 에러 메시지
- **영향**: 사용자 혼동, 디버깅 어려움
- **난이도**: ⭐⭐ 중간
- **작업 내용**:
  ```python
  class PDFProcessingError(Exception):
      """기본 예외"""
  
  class EmptyPDFError(PDFProcessingError):
      """내용이 없는 PDF"""
  
  class InsufficientChunksError(PDFProcessingError):
      """분할 가능한 텍스트 부족"""
  ```

#### 5. 배치 사이즈 자동 최적화
- **파일**: `src/rag_core.py` (L93-105)
- **문제**: 배치 사이즈 하드코딩 (64)
- **영향**: GPU 메모리 낭비 또는 OOM 위험
- **난이도**: ⭐⭐ 중간
- **작업 내용**:
  - GPU 여유 메모리 감지
  - 휴리스틱 기반 최적 배치 사이즈 계산

#### 6. 설정 검증 (Pydantic)
- **파일**: `src/config_validation.py` (새 파일)
- **문제**: 설정 값 유효성 검사 없음
- **영향**: 잘못된 설정으로 앱 크래시 가능
- **난이도**: ⭐⭐ 중간
- **작업 내용**:
  ```python
  class ModelConfig(BaseModel):
      default_ollama: str
      temperature: float = Field(ge=0, le=1)
      num_ctx: int = Field(ge=512, le=32000)
  ```

#### 7. 중복 제거 개선 (SHA256)
- **파일**: `src/graph_builder.py` (L88-105)
- **문제**: 해시 충돌 가능성
- **난이도**: ⭐⭐ 중간
- **작업 내용**:
  - hash() 대신 SHA256 사용
  - 내용 + 출처를 함께 해싱

---

### 🟢 **P3 - 장기 적용 (1-2주)**

#### 8. 기본 유닛 테스트
- **파일**: `tests/` (새 디렉토리)
- **문제**: 테스트 코드 부재
- **난이도**: ⭐⭐⭐ 높음
- **대상**:
  - `test_utils.py` - 유틸 함수 (clean_query_text, preprocess_text)
  - `test_semantic_chunker.py` - 의미론적 청킹
  - `test_rag_core_integration.py` - RAG 파이프라인

#### 9. 타입 힌트 강화
- **파일**: `src/session.py`, `src/utils.py`
- **문제**: 약한 타입 힌트 (Any 남용)
- **난이도**: ⭐⭐ 중간
- **작업 내용**:
  ```python
  T = TypeVar('T')
  
  @overload
  def get(cls, key: str) -> Any: ...
  
  @overload
  def get(cls, key: str, default: T) -> T | Any: ...
  ```

#### 10. 경쟁 조건 방지
- **파일**: `src/main.py` (L175-185)
- **문제**: TOCTOU (Time of Check, Time of Use) 문제
- **난이도**: ⭐⭐⭐ 높음
- **작업 내용**:
  ```python
  class ThreadSafeSessionManager:
      _lock = threading.RLock()
      
      def try_set_processing(self, key: str) -> bool:
          """원자적 연산"""
  ```

---

## 📊 우선순위 매트릭스

```
높음 우선순위        중간                       낮음
────────────────────────────────────────────────
P1 타임아웃         P2 오류 처리              P3 테스트
P1 상수 정의        P2 배치 최적화            P3 타입 힌트
P1 로깅 설정        P2 설정 검증              P3 경쟁 조건
                   P2 중복 제거
```

---

## 🎯 **추천 실행 순서**

### 1주차
```
[Day 1]
├─ 상수 정의 (constants.py) ✅ 1시간
├─ 로깅 설정 (logging_config.py) ✅ 1시간
└─ 타임아웃 처리 (graph_builder.py) ✅ 1시간
   └─ 테스트: 스트레스 테스트

[Day 2-3]
├─ 고급 오류 처리 (PDFProcessingError) ✅ 3시간
├─ 중복 제거 개선 (SHA256) ✅ 1시간
└─ 테스트 및 검증

[Day 4-5]
├─ 배치 사이즈 자동 최적화 ✅ 3시간
├─ 설정 검증 (Pydantic) ✅ 2시간
└─ 통합 테스트
```

### 2주차
```
[Week 2]
├─ 기본 유닛 테스트 (test_utils.py 등) ✅ 5시간
├─ 타입 힌트 강화 ✅ 3시간
└─ 경쟁 조건 방지 ✅ 4시간
```

---

## 📈 **예상 효과**

| 작업 | 개선 효과 | 사용자 영향 |
|------|---------|-----------|
| 타임아웃 처리 | 무한 대기 방지 | ⭐⭐⭐ 높음 |
| 상수 정의 | 유지보수성 ↑ | ⭐⭐ 중간 |
| 로깅 설정 | 디버깅 ↑ | ⭐⭐ 중간 |
| 오류 처리 | 사용성 ↑ | ⭐⭐⭐ 높음 |
| 배치 최적화 | 성능 ↑ | ⭐ 낮음 |
| 설정 검증 | 안정성 ↑ | ⭐⭐ 중간 |
| 테스트 | 신뢰성 ↑ | ⭐⭐⭐ 높음 |

---

## ✅ 완료된 작업

```
✅ Pickle 보안 이슈 완화
   ├─ cache_security.py (314줄)
   ├─ config.yml 확장
   ├─ rag_core.py 통합
   ├─ test_cache_security.py (620줄)
   └─ migrate_cache_v1_to_v2.py

✅ 테스트/호환성 안정화 (Windows 포함)
   ├─ pytest 전체 통과: 200 passed, 3 skipped
   ├─ Windows 환경에서 `sentence-transformers/torchvision` 로드 시 하드 크래시(0xc0000139) 가능성 확인
   │  └─ 통합 테스트에서 임베딩 로드 구간을 mock 처리하여 CI/로컬 테스트 안정화
   ├─ legacy 테스트/클라이언트 호환 shim 추가
   │  ├─ `rag_core.RAGSystem` (경량 facade)
   │  ├─ `graph_builder.build_graph()` retriever 미지정 호출 지원
   │  └─ `model_loader.load_embedding_model()` model_name 생략 허용 (config 기본값 사용)

✅ 실제 버그 수정
   ├─ `performance_monitor.py`: 메모리 델타 단위(MB) 계산 오류 수정
   ├─ `threading_safe_session.py`: Streamlit 미존재 환경에서 get_stats 안전화
   └─ `cache_security.py`: HMAC 계산 digestmod 인자 버그 수정 (Python 3.12 호환)
```

---

## 📝 상세 작업 분석

### 🔴 P1-1: 타임아웃 처리 강화

**현재 코드:**
```python
# graph_builder.py - L72-88
async def _safe_ainvoke(q):
    try:
        if hasattr(retriever, "ainvoke"):
            return await retriever.ainvoke(q)  # ❌ 타임아웃 없음
        return await asyncio.to_thread(retriever.invoke, q)
    except Exception as e:
        logger.error(f"검색 오류 ({q}): {e}")
        return []
```

**개선 후:**
```python
async def _safe_ainvoke_with_timeout(q: str, timeout: float = 30.0):
    try:
        if hasattr(retriever, "ainvoke"):
            return await asyncio.wait_for(
                retriever.ainvoke(q),
                timeout=timeout
            )
        return await asyncio.wait_for(
            asyncio.to_thread(retriever.invoke, q),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"검색 타임아웃: {timeout}초 초과 ({q})")
        return []
    except Exception as e:
        logger.error(f"검색 오류 ({q}): {e}")
        return []
```

**설정:**
```yaml
# config.yml
search:
  retriever_timeout: 30.0  # 초
  llm_timeout: 300.0       # 5분
```

---

### 🔴 P1-2: 상수 정의

**생성할 파일: `src/constants.py`**

```python
from enum import IntEnum

class UIConstants(IntEnum):
    """UI 관련 상수"""
    CONTAINER_HEIGHT = 650
    CHAT_SCROLL_HEIGHT = 650
    PDF_VIEWER_HEIGHT = 650

class PerformanceConstants(IntEnum):
    """성능 관련 상수"""
    EMBEDDING_BATCH_SIZE_DEFAULT = 64
    EMBEDDING_BATCH_SIZE_CPU = 32
    MODEL_CACHE_TTL = 600
    MAX_MESSAGE_HISTORY = 1000

class ChunkingConstants(IntEnum):
    """청킹 관련 상수"""
    MIN_CHUNK_SIZE = 200
    MAX_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_SIZE = 500
    OVERLAP_SIZE = 100
```

**사용:**
```python
# 현재
height = 650

# 개선
from constants import UIConstants
height = UIConstants.CONTAINER_HEIGHT
```

**영향받는 파일:**
- `src/ui.py`: UI_CONTAINER_HEIGHT 등
- `src/model_loader.py`: 배치 사이즈 (64)
- `src/session.py`: MAX_MESSAGE_HISTORY (1000)
- `src/rag_core.py`: 청킹 설정

---

### 🔴 P1-3: 중앙 집중식 로깅

**생성할 파일: `src/logging_config.py`**

```python
import logging
import logging.handlers
from pathlib import Path

def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """프로젝트 전체 로깅 설정"""
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (로테이션)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
```

**사용:**
```python
# src/main.py
from logging_config import configure_logging

logger = configure_logging(
    log_level="INFO",
    log_file=Path("logs/app.log")
)
```

---

### 🟡 P2-1: 고급 오류 처리

**수정할 파일: `src/exceptions.py` (새 파일)**

```python
class RAGError(Exception):
    """RAG 시스템 기본 예외"""
    pass

class PDFProcessingError(RAGError):
    """PDF 처리 중 발생하는 예외"""
    pass

class EmptyPDFError(PDFProcessingError):
    """내용이 없는 PDF"""
    def __init__(self, page_count: int = 0):
        super().__init__(
            f"PDF 파일이 비어있습니다 (페이지: {page_count}). "
            "스캔된 문서인 경우 OCR이 필요합니다."
        )

class InsufficientChunksError(PDFProcessingError):
    """분할 가능한 텍스트 부족"""
    def __init__(self, doc_count: int, chunk_count: int):
        super().__init__(
            f"청크 분할 실패: {doc_count}개 문서에서 {chunk_count}개 청크만 생성. "
            "청킹 설정을 확인하세요."
        )

class EmbeddingError(RAGError):
    """임베딩 모델 로드/실행 오류"""
    pass

class RetrievalTimeoutError(RAGError):
    """검색 타임아웃"""
    pass
```

---

## 🎬 **다음 단계**

어느 작업부터 시작하고 싶으신가요?

1. **🔴 P1 (우선순위 1)** - 빠르고 효과적
   - [ ] 상수 정의 (1시간)
   - [ ] 로깅 설정 (1시간)
   - [ ] 타임아웃 처리 (1시간)

2. **🟡 P2 (우선순위 2)** - 중요도 높음
   - [ ] 오류 처리 (3시간)
   - [ ] 설정 검증 (2시간)

3. **🟢 P3 (우선순위 3)** - 장기 계획
   - [ ] 유닛 테스트 (5시간)
   - [ ] 타입 힌트 (3시간)

**추천**: P1 먼저 완료하고, P2로 진행하세요! ✨

---

## 🧭 리뷰 기반 추가 우선순위 (현 상태 기준)

### 🔴 **P0 - 바로 처리 권장 (안정성/운영 리스크)**
- **Windows ML 스택 크래시 회피 가이드 문서화**
  - **문제**: 특정 환경에서 `torchvision`/`timm` 연동 경로가 “예외”가 아니라 “프로세스 크래시”를 유발
  - **조치**:
    - `readme.md`에 “Windows에서 embedding 모델 로드 실패/크래시” 트러블슈팅 추가
    - 필요 시 `requirements.txt`에서 Windows용 패키지 핀/옵션 분리 검토

### 🟡 **P1 - 1~2일 내 개선**
- **Pydantic v2 경고 제거**
  - **파일**: `src/cache_security.py`
  - **내용**: `@validator` → `@field_validator`로 마이그레이션(현재 테스트는 통과하지만 경고 노이즈 큼)

### 🟢 **P2 - 여유 있을 때**
- **모듈 경계 정리**
  - 테스트/레거시 호환 shim을 별도 모듈로 분리(`src/compat.py` 등)해서 core 코드의 “순수성” 유지
