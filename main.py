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
    st.Page("pages/SimpleRAG.py", title="SimpleRAG", icon=":material/chat:"),
    st.Page("pages/IndexingRAG.py", title="IndexingRAG", icon=":material/favorite:"),
])
pg.run()


