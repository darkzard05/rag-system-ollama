# API Domain

## OVERVIEW
Provides external access to RAG capabilities via a FastAPI server with REST endpoints and SSE streaming (real-time streaming is SSE-only).

## STRUCTURE
- `api_server.py`: Main FastAPI application, REST endpoints, auth bootstrap, PDF library, ownership registry.
- `schemas.py`: Pydantic models for request/response validation.
- `streaming_handler.py`: SSE (Server-Sent Events) management for real-time streaming.

## ENDPOINTS
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/login` | – | username/password → `{access_token, token_type, expires_in, session_id}` |
| POST | `/api/v1/logout` | Bearer | Revokes the presented Bearer token; optional body `session_id` |
| GET | `/api/v1/health` | – | Server status |
| POST | `/api/v1/upload` | Bearer | Upload a PDF; persists to the library and binds ownership |
| POST | `/api/v1/query` | Bearer | Non-streaming RAG query |
| POST | `/api/v1/stream_query` | Bearer | SSE RAG streaming; emits `event: error` on processing failure (VectorStoreError et al.) |
| DELETE | `/api/v1/session/{session_id}` | Bearer | Delete session data (owner only) |
| GET | `/api/v1/admin/stats` | Bearer | System + auth stats (admin role only) |
| GET | `/api/v1/pdf/{hash}` | Bearer | Serve a PDF from the library (owner only) |

## AUTH & OWNERSHIP
- **Bootstrap** (`_bootstrap_credentials`): admin account (username `admin_user`, role `admin`) created at boot. Uses `TEST_ADMIN_PASSWORD` (else a random password is generated and printed to stderr once) and `TEST_API_KEY` (else a random 24h key is generated and printed to stderr once). Credentials go to **stderr**, never the file logger.
- **State persistence**: users/sessions/API keys + a durable **deny-list** (revocation) are persisted to `AUTH_STATE_FILE` (default `.model_cache/auth_state.json`). JWT secret is persisted to `AUTH_SECRET_FILE` (default `.model_cache/.jwt_secret`, `0o600`) when `JWT_SECRET_KEY` is unset. Revocation survives restarts.
- **Ownership policy**: files **fail-closed** (unbound or foreign file → `403` on `/api/v1/pdf/{hash}`); sessions **fail-open** (unbound session accessible to any authenticated user — documented shared-legacy). Bindings are first-wins at upload start.

## PDF LIFECYCLE
- Uploads are persisted content-addressed (`{hash}.pdf`) under `PDF_STORAGE_DIR` = `data/temp/pdf_library` and retained 30 days via the retention sweep (`_sweep_expired_library_files`).
- API uploads set session `pdf_library_path`, so session cleanup does **not** delete library files. UI-session temp files (no `pdf_library_path`) are still deleted on session cleanup and are not API-serveable.

## CONVENTIONS
- Use Pydantic for all request/response validation.
- Use `StreamingResponse` for long-running RAG queries.
- All endpoints except `login`/`health` require authentication via `verify_token` (JWT or API key).

## ANTI-PATTERNS
- Do not put business logic in `api_server.py`; delegate to `core/`.
- Do not bypass `AuthenticationManager` for any endpoint.
- Avoid long-running synchronous tasks in FastAPI endpoints; use `async` and `StreamingResponse`.
