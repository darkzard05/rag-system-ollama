# API 참고 문서

## 개요

RAG System REST API는 모든 시스템 기능에 대한 프로그래밍 인터페이스를 제공합니다.

- **Base URL**: `http://localhost:8000`
- **API Version**: `v1`
- **Authentication**: JWT Bearer Token
- **Content-Type**: `application/json`

---

## 인증

### 로그인

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "password"
}
```

**응답 (200 OK):**
```json
{
  "status_code": 200,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 3600
  },
  "message": "Login successful"
}
```

### 토큰 사용

모든 요청에 Authorization 헤더 추가:

```bash
Authorization: Bearer {token}
```

### 로그아웃

```http
POST /api/v1/auth/logout
Authorization: Bearer {token}
```

---

## 검색 엔드포인트

### 1. 검색 수행

```http
POST /api/v1/search
Authorization: Bearer {token}
Content-Type: application/json

{
  "query": "주요 내용이 무엇인가?",
  "top_k": 5,
  "threshold": 0.5,
  "filters": {
    "source": "document.pdf"
  }
}
```

**요청 매개변수:**

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| query | string | ✅ | 검색 쿼리 |
| top_k | integer | ❌ | 반환할 결과 수 (기본: 5) |
| threshold | float | ❌ | 유사도 임계값 (0-1, 기본: 0.5) |
| filters | object | ❌ | 필터링 조건 |

**응답 (200 OK):**
```json
{
  "status_code": 200,
  "data": {
    "query": "주요 내용이 무엇인가?",
    "results": [
      {
        "id": 0,
        "content": "이 문서의 주요 내용은...",
        "score": 0.95,
        "source": "document.pdf",
        "page": 1,
        "chunk_index": 0
      }
    ],
    "result_count": 1,
    "execution_time_ms": 150.5
  },
  "message": "Search successful"
}
```

**에러 응답 (400 Bad Request):**
```json
{
  "status_code": 400,
  "error": "Invalid query",
  "message": "Query cannot be empty"
}
```

### 2. 검색 히스토리

```http
GET /api/v1/search/history?limit=10
Authorization: Bearer {token}
```

**응답 (200 OK):**
```json
{
  "status_code": 200,
  "data": [
    {
      "id": 1,
      "query": "주요 내용",
      "timestamp": "2025-01-21T10:30:00",
      "results_count": 3,
      "execution_time_ms": 125.5
    }
  ]
}
```

---

## 문서 관리 엔드포인트

### 1. 문서 업로드

```http
POST /api/v1/documents
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: [binary PDF file]
```

**응답 (201 Created):**
```json
{
  "status_code": 201,
  "data": {
    "document_id": "doc_123456",
    "filename": "document.pdf",
    "size_bytes": 1024000,
    "chunks_count": 25,
    "upload_time_ms": 2500,
    "status": "processed"
  },
  "message": "Document uploaded successfully"
}
```

### 2. 문서 목록 조회

```http
GET /api/v1/documents?page=1&limit=10
Authorization: Bearer {token}
```

**응답 (200 OK):**
```json
{
  "status_code": 200,
  "data": {
    "documents": [
      {
        "document_id": "doc_123456",
        "filename": "document.pdf",
        "size_bytes": 1024000,
        "chunks_count": 25,
        "upload_date": "2025-01-21T10:00:00",
        "status": "active"
      }
    ],
    "total": 1,
    "page": 1,
    "limit": 10
  }
}
```

### 3. 문서 삭제

```http
DELETE /api/v1/documents/{document_id}
Authorization: Bearer {token}
```

**응답 (200 OK):**
```json
{
  "status_code": 200,
  "data": {
    "document_id": "doc_123456",
    "status": "deleted"
  },
  "message": "Document deleted successfully"
}
```

---

## 배포 관리 엔드포인트

### 1. 배포 시작

```http
POST /api/v1/deployments
Authorization: Bearer {token}
Content-Type: application/json

{
  "service_name": "rag-service",
  "version": "1.0.0",
  "environment": "production"
}
```

**응답 (200 OK):**
```json
{
  "status_code": 200,
  "data": {
    "deployment_id": "dep_1234567890",
    "service_name": "rag-service",
    "version": "1.0.0",
    "status": "in_progress",
    "created_at": "2025-01-21T10:30:00"
  }
}
```

### 2. 배포 상태 조회

```http
GET /api/v1/deployments/{deployment_id}
Authorization: Bearer {token}
```

**응답:**
```json
{
  "status_code": 200,
  "data": {
    "deployment_id": "dep_1234567890",
    "service_name": "rag-service",
    "version": "1.0.0",
    "status": "completed",
    "progress": 100,
    "start_time": "2025-01-21T10:30:00",
    "end_time": "2025-01-21T10:35:00"
  }
}
```

### 3. 모든 배포 목록

```http
GET /api/v1/deployments
Authorization: Bearer {token}
```

---

## 캐시 관리 엔드포인트

### 1. 캐시 통계

```http
GET /api/v1/cache/stats
Authorization: Bearer {token}
```

**응답:**
```json
{
  "status_code": 200,
  "data": {
    "total_cache_size_mb": 512.5,
    "hit_rate": 0.85,
    "miss_rate": 0.15,
    "entry_count": 1000,
    "l1_cache_size_mb": 256,
    "l2_cache_size_mb": 256.5
  }
}
```

### 2. 캐시 초기화

```http
POST /api/v1/cache/clear
Authorization: Bearer {token}
```

**응답:**
```json
{
  "status_code": 200,
  "data": {
    "cleared_entries": 1000,
    "freed_memory_mb": 512.5
  },
  "message": "Cache cleared successfully"
}
```

---

## 시스템 모니터링 엔드포인트

### 1. 시스템 상태

```http
GET /api/v1/health
Authorization: Bearer {token}
```

**응답:**
```json
{
  "status_code": 200,
  "data": {
    "service_name": "rag-system",
    "version": "1.0.0",
    "status": "healthy",
    "uptime_seconds": 3600,
    "timestamp": "2025-01-21T11:30:00",
    "components": {
      "ollama": "connected",
      "vector_store": "ready",
      "cache": "operational"
    }
  }
}
```

### 2. 성능 메트릭

```http
GET /api/v1/metrics
Authorization: Bearer {token}
```

**응답:**
```json
{
  "status_code": 200,
  "data": {
    "requests_per_second": 10.5,
    "average_latency_ms": 125.3,
    "p95_latency_ms": 250.0,
    "p99_latency_ms": 500.0,
    "cache_hit_rate": 0.85,
    "error_rate": 0.01,
    "memory_usage_mb": 512,
    "cpu_usage_percent": 45.2
  }
}
```

### 3. 알림 조회

```http
GET /api/v1/notifications
Authorization: Bearer {token}
```

**응답:**
```json
{
  "status_code": 200,
  "data": [
    {
      "id": "notif_123",
      "type": "deployment_completed",
      "message": "Deployment successful",
      "severity": "info",
      "timestamp": "2025-01-21T10:35:00",
      "read": false
    }
  ]
}
```

### 4. 알림 읽음 표시

```http
POST /api/v1/notifications/{notification_id}/read
Authorization: Bearer {token}
```

---

## 에러 처리

### 에러 응답 형식

```json
{
  "status_code": 400,
  "error": "BadRequest",
  "message": "자세한 에러 메시지",
  "details": {
    "field": "specific error info"
  }
}
```

### 일반적인 에러 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 201 | 생성됨 |
| 400 | 잘못된 요청 |
| 401 | 인증 필요 |
| 403 | 권한 없음 |
| 404 | 찾을 수 없음 |
| 500 | 서버 오류 |
| 503 | 서비스 불가 |

---

## 레이트 제한

각 사용자는:
- 분당 60개 요청
- 시간당 1000개 요청

제한 초과 시 429 (Too Many Requests) 응답

---

## 예제 코드

### cURL

```bash
# 검색
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"query": "검색어", "top_k": 5}'
```

### Python

```python
import requests

token = "your-token-here"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# 검색
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={"query": "검색어", "top_k": 5},
    headers=headers
)
print(response.json())
```

### JavaScript

```javascript
const token = "your-token-here";

fetch("http://localhost:8000/api/v1/search", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    query: "검색어",
    top_k: 5
  })
})
  .then(res => res.json())
  .then(data => console.log(data));
```

---

**마지막 업데이트:** 2025-01-21
