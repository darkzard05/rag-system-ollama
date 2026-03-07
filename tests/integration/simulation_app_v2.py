import streamlit as st
import time

# [최적화 V2] 0. 초기화
if "parent_rerun_count" not in st.session_state:
    st.session_state.parent_rerun_count = 0
if "chat_rerun_count" not in st.session_state:
    st.session_state.chat_rerun_count = 0
if "pdf_rerun_count" not in st.session_state:
    st.session_state.pdf_rerun_count = 0
if "target_page" not in st.session_state:
    st.session_state.target_page = 1
if "messages" not in st.session_state:
    st.session_state.messages = []

# 부모 리런 카운트 증가
st.session_state.parent_rerun_count += 1

st.title("🚀 Fragment Sync V2 (Fixed Layout)")
st.write(f"📊 Parent Rerun Count: **{st.session_state.parent_rerun_count}**")

# --- Layout 정의 ---
col_sidebar, col_main = st.columns([1, 2])

# --- PDF Viewer Fragment 정의 ---
@st.fragment(run_every=1.0)
def pdf_viewer_fragment():
    st.session_state.pdf_rerun_count += 1
    # [핵심] 외부 컨테이너(col_sidebar)를 내부에서 호출하지 않음!
    st.subheader("📄 PDF Viewer")
    st.info(f"Current Page: **{st.session_state.target_page}**")
    st.caption(f"PDF Sync Count: {st.session_state.pdf_rerun_count}")
    
    if st.button("Manual Next", key="pdf_next_btn"):
        st.session_state.target_page += 1
        st.rerun(scope="fragment")

# --- Chat Interface Fragment 정의 ---
@st.fragment
def chat_interface_fragment():
    st.session_state.chat_rerun_count += 1
    st.subheader("💬 Chat Interface")
    st.caption(f"Chat Reruns: {st.session_state.chat_rerun_count}")
    
    # 메시지 루프
    for i, msg in enumerate(st.session_state.messages):
        st.text(f"User: {msg['text']}")
        if st.button(f"Jump to Page {msg['page']}", key=f"jump_v2_{i}"):
            st.session_state.target_page = msg['page']
            # [핵심] 부모 리런 없이 본인만 리런하거나 아예 안 함
            st.success(f"Targeting page {msg['page']}...")
            st.rerun(scope="fragment")

    user_input = st.chat_input("Ask a question...", key="chat_input_v2")
    if user_input:
        st.session_state.messages.append({
            "text": user_input, 
            "page": (len(st.session_state.messages) % 5) + 1
        })
        st.rerun(scope="fragment")

# --- 렌더링 호출 (컨테이너 내부에서 프래그먼트 실행) ---
with col_sidebar:
    pdf_viewer_fragment()

with col_main:
    chat_interface_fragment()
