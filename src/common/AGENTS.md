# AGENTS.md - src/common

## OVERVIEW
Shared utilities, schemas, and configuration management for the RAG system.

## STRUCTURE
- `config.py`: Loads application settings from `config.yml` and environment variables.
- `config_validation.py`: Pydantic-based validation for all configuration sections.
- `exceptions.py`: Custom exception hierarchy (e.g., `PDFProcessingError`, `VectorStoreError`).
- `schemas.py`: Centralized data models (Pydantic) for chat, performance, and RAG state.
- `text_utils.py`: Text processing and specialized tokenization (e.g., `bm25_tokenizer`).
- `utils.py`: General purpose utility functions.
- `constants.py`: Global constants.
- `error_handler.py`: Centralized error handling logic.
- `logging_config.py`: Standardized logging configuration.
- `circuit_breaker.py`: Resilience patterns for external calls.
- `streaming.py`: Utilities for handling asynchronous data streams.
- `typing_utils.py`: Custom type aliases and protocols.

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Add new shared exception | `src/common/exceptions.py` | Inherit from `PDFProcessingError` |
| Add new configuration field | `src/common/config_validation.py` | Update `ApplicationConfig` and sub-configs |
| Define new data schema | `src/common/schemas.py` | Use Pydantic `BaseModel` |
| Add text processing utility | `src/common/text_utils.py` | Consider tokenization impact |
| Update global constants | `src/common/constants.py` | |
| Modify logging behavior | `src/common/logging_config.py` | |

## CONVENTIONS
- Use Pydantic v2 for all schemas and config validation.
- All custom exceptions must inherit from `PDFProcessingError`.
- Mandatory type annotations for all function signatures.
- Strictly follow `ruff` formatting and linting.

## ANTI-PATTERNS
- No bare `except: pass`.
- No hardcoded configuration; use `config.py` or environment variables.
- No business logic in `src/common/`; keep it for shared utilities only.
