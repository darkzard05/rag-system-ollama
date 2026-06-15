import threading
import pytest
import sys
import os

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.session.manager import SessionManager

def test_thread_safe_global_state():
    """Verify that background threads update global store without touching streamlit state."""
    # Note: We can't easily mock st.session_state here without full streamlit context,
    # but we can verify that SessionManager.set works correctly in a thread
    # and and later verify the code changes prevent st.session_state access.
    SessionManager.init_session("test_session")
    
    def worker():
        SessionManager.set("bg_key", "bg_value", session_id="test_session")
        
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    
    # Value should be in global store
    assert SessionManager.get("bg_key", session_id="test_session") == "bg_value"
    print("\n✅ Global state updated correctly from background thread.")

if __name__ == "__main__":
    test_thread_safe_global_state()
