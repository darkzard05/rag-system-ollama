"""
Source Explorer 컴포넌트.
현재 대화에서 참조된 문서들의 리스트를 표시하고, 클릭 시 해당 페이지로 이동합니다.
"""

from typing import Any

import streamlit as st
from src.core.session import SessionManager


def render_source_explorer():
    """
    참조된 문서 리스트를 렌더링합니다.
    """
    st.subheader("📑 Source Explorer")

    current_sid = SessionManager.get_session_id()
    messages = SessionManager.get_messages(session_id=current_sid) or []

    # 1. 참조된 문서 추출 (중복 제거: file_path + page)
    referenced_docs_map: dict[tuple[str, int], dict[str, Any]] = {}

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("documents"):
            for doc in msg.get("documents"):
                # metadata에서 정보 추출
                meta = (
                    getattr(doc, "metadata", {})
                    if hasattr(doc, "metadata")
                    else doc.get("metadata", {})
                )

                file_path = meta.get("file_path")
                page = meta.get("page")
                file_name = meta.get("source") or meta.get("file_name")

                if file_path and page is not None:
                    key = (file_path, int(page))
                    if key not in referenced_docs_map:
                        # 스니펫 생성 (최대 150자)
                        content = (
                            getattr(doc, "page_content", "")
                            if hasattr(doc, "page_content")
                            else doc.get("page_content", "")
                        )
                        snippet = (
                            (content[:150] + "...") if len(content) > 150 else content
                        )

                        referenced_docs_map[key] = {
                            "file_name": file_name or "Unknown Document",
                            "page": int(page),
                            "snippet": snippet,
                            "file_path": file_path,
                        }

    if not referenced_docs_map:
        st.info("참조된 문서가 없습니다.")
        return

    # 2. 문서 리스트 렌더링
    for (file_path, page), info in referenced_docs_map.items():
        with st.container(border=True):
            # 문서 제목 및 페이지
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"**📄 {info['file_name']}**")
                st.caption(f"Page {info['page']}")

            # 스니펫 표시
            st.caption(info["snippet"])

            # 페이지 이동 버튼
            if st.button(
                f"Go to Page {info['page']}",
                key=f"source_jump_{file_path}_{page}",
                use_container_width=True,
            ):
                SessionManager.set(
                    "pdf_target_page", info["page"], session_id=current_sid
                )
                SessionManager.set("current_page", info["page"], session_id=current_sid)
                st.rerun(scope="fragment")
