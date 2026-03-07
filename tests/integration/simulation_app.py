import streamlit as st
import time

# [시뮬레이션] 0. 설정: 부모 리런 횟수와 프래그먼트 리런 횟수 추적
if "parent_rerun_count" not in st.session_state:
    st.session_state.parent_rerun_count = 0
if "chat_rerun_count" not in st.session_state:
    st.session_state.chat_rerun_count = 0
if "pdf_rerun_count" not in st.session_state:
    st.session_state.pdf_rerun_count = 0

# 1. 상태 전이를 위한 공통 상태
if "target_page" not in st.session_state:
    st.session_state.target_page = 1
if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.parent_rerun_count += 1

st.title("Fragment Sync Simulation")
st.write(f"📊 Parent Rerun Count: **{st.session_state.parent_rerun_count}**")

# --- Layout ---
col_sidebar, col_main = st.columns([1, 2])

# --- PDF Viewer Fragment ---
@st.fragment
def pdf_viewer_fragment():
    st.session_state.pdf_rerun_count += 1
    with col_sidebar:
        st.subheader("📄 PDF Viewer (Fragment)")
        st.info(f"Current Page: **{st.session_state.target_page}**")
        st.write(f"PDF Reruns: {st.session_state.pdf_rerun_count}")
        
        # 페이지 직접 이동 버튼
        if st.button("Manual Next Page"):
            st.session_state.target_page += 1
            # fragment 내부에서 st.rerun()은 이 fragment만 다시 그림
            st.rerun()

# --- Chat Interface Fragment ---
@st.fragment
def chat_interface_fragment():
    st.session_state.chat_rerun_count += 1
    with col_main:
        st.subheader("💬 Chat (Fragment)")
        st.write(f"Chat Reruns: {st.session_state.chat_rerun_count}")
        
        # 메시지 표시
        for i, msg in enumerate(st.session_state.messages):
            st.text(f"User: {msg['text']}")
            if st.button(f"Jump to Page {msg['page']}", key=f"jump_{i}"):
                # [핵심] 여기서 페이지 이동 시도
                st.session_state.target_page = msg['page']
                # 우려사항: 여기서 st.rerun()을 안 하면 PDF 뷰어가 안 바뀌고, 
                # st.rerun()을 하면 전체 페이지가 깜빡임.
                
                # 해결책: 부모(전체) 리런 없이 PDF 프래그먼트만 업데이트할 수 있는가?
                # Streamlit의 한계: 프래그먼트 A에서 프래그먼트 B를 직접 리런시킬 순 없음.
                # 단, 부모가 리런되면 둘 다 새로 그려짐.
                st.warning("Navigating to page... (Requires Parent Rerun to see PDF update)")
                st.rerun() # 현재 로직은 전체 리런 유도 (잔상의 원인)

        # 질문 입력
        user_input = st.chat_input("Ask a question...")
        if user_input:
            # 새 메시지 추가 (랜덤 페이지 할당)
            st.session_state.messages.append({"text": user_input, "page": (len(st.session_state.messages) % 5) + 1})
            st.rerun()

# --- 렌더링 호출 ---
pdf_viewer_fragment()
chat_interface_fragment()

st.divider()
st.markdown("""
### 🧪 테스트 시나리오 분석
1. **채팅 입력**: 채팅창만 리런되어야 함 (Parent Rerun Count가 올라가면 실패).
2. **페이지 점프**: 버튼 클릭 시 PDF 뷰어의 `Current Page`가 즉시 바뀌어야 함.
3. **완성도 평가**: 페이지 점프 시 Parent Rerun이 발생하는지, 아니면 PDF Fragment만 스마트하게 바뀌는지 확인.
""")
