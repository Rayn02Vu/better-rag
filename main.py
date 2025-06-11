import streamlit as st
from streamlit import session_state as ss

if "messages" not in ss:
    ss.messages = []
if "loading" not in ss:
    ss.loading = False
if "prompt" not in ss:
    ss.prompt = ""

pg = st.navigation([
    st.Page("pages/Home.py", title="Home", icon=":material/home:"),
    st.Page("pages/VectorPreview.py", title="VectorPreview", icon=":material/bolt:"),
    st.Page("pages/QueryIndex.py", title="QueryIndex", icon=":material/favorite:"),
    st.Page("pages/SimpleRAG.py", title="SimpleRAG", icon=":material/manage_search:"),
    st.Page("pages/FusionRAG.py", title="FusionRAG", icon=":material/flash_on:"),
    st.Page("pages/AdIndexing.py", title="AdIndexing", icon=":material/select_all:")
])
pg.run()


