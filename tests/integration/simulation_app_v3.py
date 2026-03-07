import streamlit as st

# [V3 시뮬레이션] 0. 초기화
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

# 부모 리런 카운트 (이게 1로 유지되어야 성공)
st.session_state.parent_rerun_count += 1

st.title("V3: True Fragment Isolation")
st.write(f"📊 Parent Reruns: {st.session_state.parent_rerun_count}")

# 1. PDF 영역 (프래그먼트)
@st.fragment(run_every=2.0)
def pdf_area():
    st.session_state.pdf_rerun_count += 1
    with st.container(border=True):
        st.subheader("📄 PDF Viewer")
        st.info(f"Page: {st.session_state.target_page}")
        st.caption(f"PDF Reruns: {st.session_state.pdf_rerun_count}")

# 2. 채팅 영역 (프래그먼트)
@st.fragment
def chat_area():
    st.session_state.chat_rerun_count += 1
    with st.container(border=True):
        st.subheader("💬 Chat")
        st.write(f"Chat Reruns: {st.session_state.chat_rerun_count}")
        
        # 메시지 및 점프 버튼
        for i, m in enumerate(st.session_state.messages):
            if st.button(f"Jump to {m['page']}", key=f"btn_v3_{i}"):
                st.session_state.target_page = m['page']
                # [중요] 여기서 st.rerun()을 호출하면 '부모'가 리런됨.
                # 하지만 이 시뮬레이션의 목적은 "채팅 입력" 시 부모가 안 흔들리는지 보는 것.

        # 입력창 (key를 명확히 지정)
        if prompt := st.chat_input("Input...", key="input_v3"):
            st.session_state.messages.append({"text": prompt, "page": 5})
            # 채팅 입력 시 st.rerun() 호출 (이것은 현재 프래그먼트만 다시 그림)
            st.rerun()

# 호출
pdf_area()
chat_area()
