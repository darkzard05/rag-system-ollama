# Design Spec: SessionManager Refactoring (Service Layer Decomposition)

- **Date**: 2026-06-01
- **Status**: Draft
- **Topic**: Refactoring the monolithic `SessionManager` into a decoupled service-based architecture.

## 1. Overview
Current `SessionManager` is a "God Object" that mixes data storage, Streamlit UI synchronization, context management, and task logging. This tight coupling makes the core RAG logic dependent on the Streamlit framework, hindering testability and portability.

This design decomposes the functionality into four distinct, specialized modules.

## 2. Goals
- **Decoupling**: Remove Streamlit dependencies from core data storage.
- **Portability**: Allow RAG core to run without a Streamlit environment.
- **Robustness**: Ensure reliable context (Session ID) propagation across threads and async tasks.
- **Maintainability**: Clear boundaries for data CRUD, UI sync, and task management.

## 3. Component Architecture

### 3.1 SessionStore (`core/session/store.py`)
Pure Python data layer for session-specific state.
- **Responsibilities**: CRUD for session data, thread-safe access, session cleanup.
- **Dependency**: None (Pure Python).
- **Key Methods**:
    - `get(key, session_id, default)`
    - `set(key, value, session_id)`
    - `delete(key, session_id)`
    - `clear(session_id)`

### 3.2 ContextManager (`core/session/context.py`)
Infrastructure layer for tracking and propagating Session IDs.
- **Responsibilities**: Bind Session IDs to execution contexts (threads/async tasks).
- **Mechanism**: `contextvars.ContextVar`.
- **Key Features**:
    - `get_current_session_id()`: Auto-extracts ID from context.
    - `@with_session_context`: Decorator for thread/task isolation.

### 3.3 UIBridge (`ui/bridge.py`)
Presentation layer connecting the Store to Streamlit.
- **Responsibilities**: Bi-directional sync between `SessionStore` and `st.session_state`.
- **Dependency**: `streamlit`, `SessionStore`.
- **Key Methods**:
    - `sync_store_to_ui()`: Bulk mirror store to UI state.
    - `sync_ui_to_store(key)`: Reflect UI changes back to store.

### 3.4 TaskService (`infra/task_service.py`)
Utility layer for asynchronous logging and pending task management.
- **Responsibilities**: Recording status logs, managing UI-thread task queues (e.g., "rebuild index").
- **Key Methods**:
    - `add_status_log(message, session_id)`
    - `push_pending_task(type, payload)`
    - `pop_pending_tasks()`

## 4. Data Flow
1. **User Action**: UI -> `UIBridge` -> `SessionStore`.
2. **Background Process**: `RAGCore` -> `ContextManager` (id) -> `TaskService` (log) -> `SessionStore` (result).
3. **UI Update**: `UIBridge` polls `TaskService`/`SessionStore` -> `st.session_state` -> Streamlit Rerender.

## 5. Implementation Strategy
1. Create new modules under `src/core/session/`, `src/ui/`, and `src/infra/`.
2. Migrating data from existing `SessionManager` in stages.
3. Update `main.py` and `rag_core.py` to use the new specialized services.
4. Final cleanup: Remove legacy `SessionManager` and `main.py` monkey patches.

## 6. Testing Plan
- **Unit Tests**: Test `SessionStore` and `TaskService` without Streamlit.
- **Context Tests**: Verify Session ID propagation in multi-threaded scenarios.
- **Integration Tests**: Verify UI synchronization using mocks for Streamlit state.
