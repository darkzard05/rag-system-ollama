import streamlit as st
import time

# [최적화 시뮬레이션] 0. 설정
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

# 부모 리런 횟수 증가 (최초 로딩 시에만 발생해야 함)
st.session_state.parent_rerun_count += 1

st.title("🚀 Optimized Fragment Sync")
st.write(f"📊 Parent Rerun Count: **{st.session_state.parent_rerun_count}**")

# --- Layout ---
col_sidebar, col_main = st.columns([1, 2])

# --- PDF Viewer Fragment ---
# [최적화] run_every를 통해 주기적으로 상태를 체크하거나, 외부 트리거를 감지함
@st.fragment(run_every=1.0) # 1초마다 동기화 (네트워크 비용 거의 없음)
def pdf_viewer_fragment_optimized():
    st.session_state.pdf_rerun_count += 1
    with col_sidebar:
        st.subheader("📄 PDF (Optimized)")
        # [핵심] 세션 상태의 target_page를 실시간 반영
        st.info(f"Current Page: **{st.session_state.target_page}**")
        st.caption(f"PDF Sync Count: {st.session_state.pdf_rerun_count}")
        
        # 내부 동작 시에는 부모 리런 없음
        if st.button("Manual Next"):
            st.session_state.target_page += 1
            # fragment 내부 전용 rerun
            st.rerun(scope="fragment")

# --- Chat Interface Fragment ---
@st.fragment
def chat_interface_fragment_optimized():
    st.session_state.chat_rerun_count += 1
    with col_main:
        st.subheader("💬 Chat (Optimized)")
        
        # 메시지 루프
        for i, msg in enumerate(st.session_state.messages):
            st.text(f"User: {msg['text']}")
            # [핵심] 인용구 클릭 시 부모 리런을 하지 않음!
            if st.button(f"Jump to Page {msg['page']}", key=f"jump_opt_{i}"):
                st.session_state.target_page = msg['page']
                st.success(f"Target page set to {msg['page']}. PDF will sync shortly.")
                # st.rerun() 을 호출하지 않아도 pdf_viewer의 run_every가 이를 감지함
                # 또는 st.rerun(scope="fragment") 만 호출하여 채팅창만 갱신

        user_input = st.chat_input("Ask a question...", key="chat_input_opt")
        if user_input:
            st.session_state.messages.append({"text": user_input, "page": (len(st.session_state.messages) % 5) + 1})
            # 채팅 입력 시에도 부모 리런 최소화 시도
            st.rerun(scope="fragment")

# --- 렌더링 호출 ---
pdf_viewer_fragment_optimized()
chat_interface_fragment_optimized()

st.divider()
st.markdown("""
### 🧪 최적화 성공 포인트
1. **Parent Rerun Count 고정**: 채팅 입력이나 페이지 점프 시에도 숫자가 올라가지 않아야 함 (부모 화면 유지).
2. **비동기 동기화**: `run_every` 또는 `scope="fragment"`를 통해 다른 영역의 변화를 즉시 또는 수초 내에 반영.
3. **잔상 제거**: 사이드바가 아예 다시 그려지지 않으므로, PDF 뷰어의 깜빡임이 0이 됨.
""")
