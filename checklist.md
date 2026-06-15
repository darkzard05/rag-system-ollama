# Checklist: Session & Loop Integrity Fix

- [x] **Task 1: Foundation - Thread-Safe Session Manager Refactoring**
    - [x] Step 1: Write a test to reproduce the thread-unsafe `set` behavior
    - [x] Step 2: Run the test to verify current behavior
    - [x] Step 3: Refactor `SessionManager.set`
    - [x] Step 4: Update `sync_to_streamlit`
    - [x] Step 5: Commit
- [x] **Task 2: Loop Integrity - RAG Core Engine Factory**
    - [x] Step 1: Write a test to reproduce the loop mismatch error
    - [x] Step 2: Run the test to verify it fails
    - [x] Step 3: Refactor `_get_rag_engine`
    - [x] Step 4: Run tests to verify fix
    - [x] Step 5: Commit
- [x] **Task 3: UI Integration and Verification**
    - [x] Step 1: Ensure `sync_to_streamlit` is called at the start of `main()`
    - [x] Step 2: Verify `astream_events` uses the updated engine factory
    - [x] Step 3: Run full integration test suite
    - [x] Step 4: Commit
