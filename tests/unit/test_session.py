import pytest
from session import SessionManager
import streamlit as st

# Since we are testing the SessionManager, we can't use the autouse fixture
# from conftest.py. We need to manually manage the session state.


@pytest.fixture(autouse=True)
def clear_session_state():
    """Clear the session state before and after each test."""
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_init_session():
    """Tests if the session is initialized correctly."""
    # 1. Arrange: The session is empty

    # 2. Act: Initialize the session
    SessionManager.init_session()

    # 3. Assert: Check if the default values are set
    assert SessionManager.get("_initialized") is True
    assert SessionManager.get("messages") == []
    assert SessionManager.get("last_selected_model") is None
    assert SessionManager.get("is_first_run") is True

    # Check if calling init_session again doesn't change the state
    SessionManager.set("is_first_run", False)
    SessionManager.init_session()
    assert SessionManager.get("is_first_run") is False


def test_reset_for_new_file():
    """Tests if the session is reset correctly for a new file."""
    # 1. Arrange: Set some RAG-related state
    SessionManager.init_session()
    SessionManager.set("pdf_processed", True)
    SessionManager.set("qa_chain", "dummy_chain")
    SessionManager.set("vector_store", "dummy_store")
    SessionManager.set("needs_rag_rebuild", False)

    # 2. Act: Reset the session for a new file
    SessionManager.reset_for_new_file()

    # 3. Assert: Check if the RAG-related state is reset
    assert SessionManager.get("pdf_processed") is False
    assert "qa_chain" not in st.session_state
    assert "vector_store" not in st.session_state
    assert SessionManager.get("needs_rag_rebuild") is True


def test_add_message():
    """Tests if messages are added correctly."""
    # 1. Arrange: Initialize the session
    SessionManager.init_session()

    # 2. Act: Add a message
    SessionManager.add_message("user", "Hello")

    # 3. Assert: Check if the message is in the session
    messages = SessionManager.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"


def test_is_ready_for_chat():
    """Tests the is_ready_for_chat method."""
    # 1. Arrange: Initialize the session
    SessionManager.init_session()

    # 2. Assert: Check initial state
    assert SessionManager.is_ready_for_chat() is False

    # 3. Act & Assert: Set pdf_processed to True
    SessionManager.set("pdf_processed", True)
    assert SessionManager.is_ready_for_chat() is False

    # 4. Act & Assert: Set qa_chain
    SessionManager.set("qa_chain", "dummy_chain")
    assert SessionManager.is_ready_for_chat() is True

    # 5. Act & Assert: Set pdf_processing_error
    SessionManager.set("pdf_processing_error", "some error")
    assert SessionManager.is_ready_for_chat() is False


def test_reset_all_state():
    """Tests if the entire session state is reset."""
    # 1. Arrange: Set some state
    SessionManager.init_session()
    SessionManager.set("is_first_run", False)
    SessionManager.set("last_selected_model", "dummy_model")

    # 2. Act: Reset all state
    SessionManager.reset_all_state()

    # 3. Assert: Check if the state is reset to default
    assert SessionManager.get("_initialized") is True
    assert SessionManager.get("is_first_run") is True
    assert SessionManager.get("last_selected_model") is None
