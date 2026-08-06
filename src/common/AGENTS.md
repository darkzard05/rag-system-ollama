# AGENTS.md - src/common

## OVERVIEW
Shared utilities, configuration management, and resilience patterns for the RAG system.

## STRUCTURE
- `config.py`: Loads application settings from `config.yml` and environment variables.
- `exceptions.py`: Custom exception hierarchy (e.g., `PDFProcessingError`, `VectorStoreError`).
- `text_utils.py`: Text processing and specialized tokenization (e.g., `bm25_tokenizer`).
- `utils.py`: General purpose utility functions (hashing, background workers, cache helpers).
- `constants.py`: Global constants.
- `logging_config.py`: Standardized logging configuration.
- `circuit_breaker.py`: Resilience patterns for external calls.
- `async_worker.py`: Async task executor replacing `nest_asyncio`.

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Add new shared exception | `src/common/exceptions.py` | Inherit from `PDFProcessingError` |
| Add new configuration field | `src/common/config.py` | Load via `config.yml` key or env var |
| Add text processing utility | `src/common/text_utils.py` | Consider tokenization impact |
| Update global constants | `src/common/constants.py` | |
| Modify logging behavior | `src/common/logging_config.py` | |
| Add resilience pattern | `src/common/circuit_breaker.py` | |
| Run async background work | `src/common/async_worker.py` | Prefer `AsyncWorker` over raw threads |

## CONVENTIONS
- All custom exceptions must inherit from `PDFProcessingError`.
- Mandatory type annotations for all function signatures.
- Strictly follow `ruff` formatting and linting.

## ANTI-PATTERNS
- No bare `except: pass`.
- No hardcoded configuration; use `config.py` or environment variables.
- No business logic in `src/common/`; keep it for shared utilities only.
