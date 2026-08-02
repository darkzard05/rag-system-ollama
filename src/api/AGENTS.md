# API Domain

## OVERVIEW
Provides external access to RAG capabilities via FastAPI, WebSockets, and SSE streaming.

## STRUCTURE
- `api_server.py`: Main FastAPI application and REST endpoints.
- `schemas.py`: Pydantic models for request/response validation.
- `streaming_handler.py`: SSE (Server-Sent Events) management for real-time streaming.
- `websocket_handler.py`: WebSocket connection and message management.

## WHERE TO LOOK
- Add/modify REST endpoints $\rightarrow$ `api_server.py`
- Update request/response models $\rightarrow$ `schemas.py`
- Adjust streaming/SSE logic $\rightarrow$ `streaming_handler.py`
- Implement/fix WebSocket messaging $\rightarrow$ `websocket_handler.py`
- Authentication/Security logic $\rightarrow$ `security/auth_system.py`

## CONVENTIONS
- Use Pydantic for all request/response validation.
- Use `StreamingResponse` for long-running RAG queries.
- WebSocket messages must follow the `WSMessage` structure.
- All endpoints require authentication via `verify_token` or API key.

## ANTI-PATTERNS
- Do not put business logic in `api_server.py`; delegate to `core/`.
- Do not bypass `AuthenticationManager` for any endpoint.
- Avoid long-running synchronous tasks in FastAPI endpoints; use `async` and `StreamingResponse`.
