"""
Test: Does st.container(height=100) create stVerticalBlockBorderWrapper?
Compare: border=True vs border=False vs default.
"""
import streamlit as st

st.set_page_config(layout="wide")

st.title("Container Border Test")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### border=True")
    with st.container(height=200, border=True):
        st.write("This container has border=True")

with col2:
    st.markdown("### border=False")
    with st.container(height=200, border=False):
        st.write("This container has border=False")

with col3:
    st.markdown("### Default")
    with st.container(height=200):
        st.write("This container uses default border")
