# RAG System Test Infrastructure

This directory contains the test suite for the RAG System. To ensure stability and maintainability, we use a behavior-driven testing approach with a centralized mock infrastructure.

## 🏗️ Architecture

### 1. Shared Fixtures (`tests/conftest.py`)
- `session_context`: Generates a unique `session_id` for each test and ensures `SessionManager.delete_session` is called after the test.

## 🧪 How to write tests

### Unit Tests (`tests/unit/`)
Focus on individual components. Use mocks to isolate the unit under test.
```python
@pytest.mark.asyncio
async def test_my_feature(session_context):
    rag = RAGSystem(session_id=session_context)
    with patch("core.some_func", return_value=mock_llm()):
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
