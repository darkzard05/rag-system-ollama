# RAG System Test Infrastructure

This directory contains the test suite for the RAG System. To ensure stability and maintainability, we use a behavior-driven testing approach with a centralized mock infrastructure.

## 🏗️ Architecture

### 1. Mock Factory (`tests/utils/mock_factory.py`)
Instead of repeating `MagicMock` setups in every test, use the factory to create standardized mock objects:
- `create_mock_llm()`: Returns an `AsyncMock` LLM with a standard RAG response structure.
- `create_mock_embedder()`: Returns a mock embedder with fixed vector outputs.
- `create_mock_vector_store()`: Returns a mock vector store with predefined similarity search results.
- `create_mock_document()`: Generates standard `Document` objects.

### 2. Shared Fixtures (`tests/conftest.py`)
- `session_context`: Generates a unique `session_id` for each test and ensures `SessionManager.delete_session` is called after the test.
- `reset_model_manager`: Resets the `ModelManager` singleton state (instances, clients) before and after each test to ensure isolation.
- `mock_rag_system`: Provides a `RAGSystem` instance configured for the current session.

### 3. Session Helpers (`tests/utils/session_helper.py`)
- `with_session(session_id)`: Context manager to switch the current active session.
- `assert_session_value(key, expected, session_id)`: Utility to verify session state.

## 🧪 How to write tests

### Unit Tests (`tests/unit/`)
Focus on individual components. Use the `mock_factory` to isolate the unit under test.
```python
@pytest.mark.asyncio
async def test_my_feature(session_context):
    rag = RAGSystem(session_id=session_context)
    # Use mock_factory for dependencies
    with patch("core.some_func", return_value=mock_factory.create_mock_llm()):
        result = await rag.my_method()
        assert result == "expected"
```

### Integration Tests (`tests/integration/`)
Focus on end-to-end flows and defect verification.
- **Defect Verification**: Check `tests/integration/test_defects_verification.py` for examples of how to verify critical fixes (e.g., memory leaks, race conditions).

## 🚀 Running Tests

Run all tests:
```bash
pytest tests/unit tests/integration
```

Run specific test file:
```bash
pytest tests/unit/test_rag_pipeline.py
```
