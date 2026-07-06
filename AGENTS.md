# Agent Guidelines for GraphRAG-Ollama

This document provides the necessary context, commands, and style guidelines for AI agents operating within the `rag-system-ollama` repository.

## 🚀 Quick Start Commands

### Development & Execution
- **Run Application**: `streamlit run src/main.py`
- **Install Dependencies**: `pip install -r requirements.txt`

### Quality Assurance
- **Linting**: `ruff check .`
- **Formatting**: `ruff format .`
- **Type Checking**: `mypy src`
- **Run All Unit Tests**: `pytest tests/unit`
- **Run Single Test**: `pytest tests/unit/test_file.py::test_function_name`
- **Integration Test**: `python scripts/test_full_pipeline.py`
- **Metadata Verification**: `python scripts/verify_section_metadata.py`

---

## 🏗️ Project Structure

- `src/main.py`: Entry point for the Streamlit application.
- `src/core/`: Core RAG logic.
    - `rag_core.py`: Main RAG orchestration and pipeline.
    - `session/`: Session state and context management.
    - `chunking.py` & `semantic_chunker.py`: Document splitting logic.
    - `graph_builder.py`: Knowledge graph construction.
- `src/ui/`: Frontend components and layout.
    - `ui.py`: Main UI orchestration.
    - `components/`: Reusable Streamlit UI elements (chat, sidebar, etc.).
- `src/api/`: API server and communication handlers (WebSockets/Streaming).
- `src/common/`: Shared utilities, configuration, and exception definitions.
- `src/cache/`: Caching layers for embeddings, coordinates, and LLM responses.
- `src/infra/`: System-level infrastructure (deployments, notifications, task service).
- `src/security/`: Authentication, RBAC, and encryption utilities.
- `src/services/`: Specialized services (monitoring, optimization, distributed processing).
- `tests/`: Test suites (unit and integration).
- `scripts/`: Utility scripts for verification and end-to-end testing.
- `config.yml`: Central configuration for models, prompts, and system parameters.

---

## 🛠️ Code Style Guidelines

### 1. General Principles
- **Python Version**: 3.10+
- **Formatting**: Strictly adhere to `ruff` formatting (Double quotes, space indentation, 88 characters line length).
- **Documentation**: Use clear docstrings for all public classes and methods.

### 2. Imports & Organization
- **Sorting**: All imports must be sorted via `ruff` (isort).
- **First-Party Modules**: The following are recognized as internal packages:
  - `api`, `common`, `core`, `infra`, `security`, `services`, `ui`
- **Lazy Imports**: Use lazy imports (inside functions) for heavy modules (e.g., `torch`, `fitz`) to minimize startup time, especially in `src/main.py` and `src/core/rag_core.py`.

### 3. Typing & Naming
- **Type Annotations**: Mandatory for all function signatures. Use `mypy` for verification.
- **Type Syntax**: Prefer `X | Y` (Python 3.10+) over `Union[X, Y]`. `Optional[X]` is permitted for backward compatibility.
- **Naming Conventions**:
  - **Classes**: `PascalCase` (e.g., `RAGSystem`, `SessionManager`, `DocumentProcessor`)
  - **Functions/Variables**: `snake_case` (e.g., `build_pipeline`, `current_file_path`, `process_document`)
  - **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_EMBEDDING_MODEL`, `MAX_CHUNK_SIZE`)
  - **Private Members**: Prefix with `_` (e.g., `_ensure_session_context`, `_load_config`)

### 4. Error Handling
- **Custom Exceptions**: Use defined exceptions in `src/common/exceptions.py` (e.g., `VectorStoreError`, `EmptyPDFError`).
- **Approach**: Avoid bare `except: pass`. Use specific exception catching and log errors via the system logger.
- **Graceful Failure**: Ensure UI components handle exceptions without crashing the Streamlit session.

### 5. Architecture Patterns
- **Session Management**: Use `core.session.SessionManager` for all state tracking to ensure thread-safety and persistence across Streamlit reruns.
- **RAG Orchestration**: Logic for the RAG pipeline must reside in `src/core/` using `LangGraph` for orchestration.
- **UI Components**: Keep UI logic in `src/ui/` and separate it from core business logic.
- **Configuration**: All tunable parameters (prompts, model names, chunk sizes) must be externalized in `config.yml`.

---

## 🧪 Testing Strategy
- **Unit Tests**: Located in `tests/unit/`. Focus on isolated logic (chunking, metadata extraction).
- **Integration Tests**: Located in `tests/integration/` or `scripts/`. Focus on the end-to-end flow from PDF upload to QA.
- **Async Testing**: Use `pytest-asyncio` with `asyncio_mode = "auto"` (configured in `pyproject.toml`).

---

## 🔄 Feature Implementation Workflow

When adding a new feature or fixing a bug, follow these steps:
1. **Research**: Explore relevant files and understand existing patterns.
2. **Plan**: Outline the changes and identify impacted modules.
3. **Implement**:
    - Add/update schemas in `src/common/schemas.py` or `src/api/schemas/`.
    - Implement core logic in `src/core/` or `src/services/`.
    - Update UI components in `src/ui/components/` if necessary.
4. **Verify**:
    - Run `ruff check .` and `ruff format .`.
    - Run `mypy src` to ensure type safety.
    - Write and run unit tests in `tests/unit/`.
    - Run integration tests via `scripts/test_full_pipeline.py`.
5. **Final Check**: Ensure no secrets are logged and Streamlit performance is maintained.

---

## 🤖 AI Coding Rules
- **언어 제약**: 사용자에 대한 모든 응답은 반드시 한국어로만 작성해야 합니다. (예외 없음)
- **Incremental Work**: Work incrementally. Read the file first, output the plan, write the changes, and run the command in separate turns.
- **Single-Task Turns**: Do not attempt to complete multiple file modifications and tests in a single turn.
- **Concise Reasoning**: Do not write complete source code blocks or large diffs inside your thinking/reasoning space. Keep thinking brief (maximum 3-5 sentences) and transition immediately to tool calls.
- **Atomic Edits**: Perform edits in the smallest possible logical units (e.g., single functions) to avoid token limits and output interruptions.
- **Thinking-Edit Separation**: Use the thinking space ONLY for high-level logic planning; avoid duplicating full code blocks in thinking if they are already in the `edit` tool call. **CRITICAL: Never attempt to modify code within the thinking block. All code changes MUST be executed using the provided file modification tools.**
- **Read-Before-Edit**: Always perform a fresh `read` of the target file immediately before an `edit` to ensure `oldString` matches the current state exactly. If an edit fails, re-read the file and provide a more unique `oldString` (including surrounding lines) to avoid ambiguity or mismatch.
- **Minimize Redundancy**: Avoid replacing large blocks of code when a small, precise change suffices.

---

## 🔄 AI Generation & Loop Control

### 1. Loop Prevention (Agent Level)
- **Max-Tries Limit**: If the same tool call or environment setup fails 3+ times, stop immediately. Do not repeat the same method; report the failure to the orchestrator and request an alternative approach.
- **Structured Attempt Log**: For complex setup tasks, maintain an internal log of `[Attempt Count / Method / Result / Root Cause]` to track progress and avoid repeating failed paths.
- **State-Gate Principle**: Follow the "Try $\rightarrow$ Verify $\rightarrow$ Proceed" flow. Never assume a command succeeded; verify the state change before moving to the next step.

### 2. Loop Recovery (Orchestrator Level)
- **Repetition Monitoring**: If a sub-agent's response shows the same phrase or tool-call pattern 3+ times, define it as a 'Generation Loop' and terminate the task immediately.
- **Context Flush & Strategy Shift**: When a loop is detected, do not reuse the failing context. Inject a "Loop Detected" alert and force the agent to adopt a completely different strategy.
- **Text-to-Tool Ratio Check**: Be wary of responses with excessive text but zero `tool_use`. This often indicates the model is "hallucinating a solution" or stuck in a reasoning loop.

---

## ⚠️ Critical Constraints
- **Zero-Error Policy**: No code should be committed if `ruff` or `mypy` reports errors.
- **No Secret Leaks**: Never commit API keys or credentials. Use environment variables or `config.yml` (which should be git-ignored in production).
- **Streamlit Performance**: Minimize `st.rerun()` calls and use `st.fragment` or `st.empty` to prevent UI flickering.
