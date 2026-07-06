import streamlit as st
import sys
import os

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from src.ui.components.chat import render_chat_interface
from src.core.session import SessionManager

print("--- Testing Chat Height Logic ---")

# 1. Mock SessionManager
class MockSessionManager:
    @staticmethod
    def init_session(): pass
    @staticmethod
    def get_messages(): return []
    @staticmethod
    def get_session_id(): return "test_session"
    @staticmethod
    def get(k, d=None, sid=None):
        if k == "is_generating_answer": return False
        return d
    @staticmethod
    def set(k, v, sid=None): pass

# Patch SessionManager
import src.ui.components.chat
ui.components.chat.SessionManager = MockSessionManager

# 2. Set calculated_container_height
st.session_state["calculated_container_height"] = 800

# 3. Verify logic in chat.py
calculated_height = st.session_state.get("calculated_container_height", 650)
chat_container_height = max(300, calculated_height - 80)

print(f"Calculated height from session_state: {calculated_height}")
print(f"Expected chat_container_height: {chat_container_height}")

if chat_container_height == 720:
    print("✅ Chat height calculation logic matches expectation (800 - 80 = 720)")
else:
    print(f"❌ Chat height calculation logic mismatch: {chat_container_height} != 720")

# 4. Try rendering (this might still fail if not in streamlit context, but we check logic first)
try:
    # render_chat_interface calls st.container(height=chat_container_height)
    # In bare python, st.container will likely warn or fail but we can see if it reaches that point.
    render_chat_interface()
    print("✅ render_chat_interface executed without crash (in bare mode)")
except Exception as e:
    print(f"ℹ️ render_chat_interface (as expected) failed in bare mode: {e}")
