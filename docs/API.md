# API Reference

This document provides the technical specifications for the RAG System API.

## 🚀 Overview

The RAG System provides a production-ready REST API built with FastAPI, allowing for document indexing and retrieval-augmented generation.

- **Base URL**: `http://localhost:8000`
- **Version**: `v1`
- **Format**: `application/json`
- **Recommended Model**: `qwen3:4b-instruct-2507-q4_K_M`

---

## 📡 Endpoints

### 1. System Health
Verify the server status and the current model being used.

- **Method**: `GET`
- **URL**: `/api/v1/health`
- **Response (200 OK)**:
    ```json
    {
      "status": "healthy",
      "timestamp": 1737528000.0,
      "model": "qwen3:4b-instruct-2507-q4_K_M"
    }
    ```

### 2. Document Upload
Upload a PDF file to the system for analysis and indexing.

- **Method**: `POST`
- **URL**: `/api/v1/upload`
- **Body**: `multipart/form-data`
    - `file`: (Binary PDF file)
- **Response (200 OK)**:
    ```json
    {
      "message": "✅ Document processing completed.",
      "filename": "report.pdf",
      "chunks_processed": 142,
      "cache_used": false
    }
    ```

### 3. Query (Synchronous)
Get a complete answer for a specific query in a single response.

- **Method**: `POST`
- **URL**: `/api/v1/query`
- **Body**:
    ```json
    {
      "query": "What is the conclusion of the report?",
      "use_cache": true
    }
    ```
- **Response (200 OK)**:
    ```json
    {
      "answer": "The report concludes that...",
      "thought": "The user is asking for a summary...",
      "sources": [{"page": 1, "content": "..."}],
      "execution_time_ms": 1250.5
    }
    ```

### 4. Stream Query (SSE)
Get real-time token streaming using Server-Sent Events (SSE).

- **Method**: `POST`
- **URL**: `/api/v1/stream_query`
- **Body**: Same as Query.
- **Data Flow**:
    - `event: thinking`: (Optional) Internal reasoning process tokens.
    - `event: message`: Real-time response tokens.
    - `event: sources`: JSON data containing retrieved document references.
    - `event: end`: Completion signal.

---

## 💻 Example Client (Python)

```python
import requests
import json

payload = {"query": "Summarize the document."}
with requests.post("http://localhost:8000/api/v1/stream_query", json=payload, stream=True) as res:
    for line in res.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                print(decoded_line[6:])
```

---

**Last Updated:** 2026-01-27
