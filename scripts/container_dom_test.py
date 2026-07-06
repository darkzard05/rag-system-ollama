"""Minimal Streamlit app to examine st.container() DOM structure."""
import streamlit as st

st.set_page_config(layout="wide")
st.title("Container DOM Structure Test")

with st.container(height=200):
    st.markdown("**Default border** container (height=200)")

with st.container(height=200, border=True):
    st.markdown("**border=True** container (height=200)")

with st.container(height=200, border=False):
    st.markdown("**border=False** container (height=200)")
