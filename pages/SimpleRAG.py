import streamlit as st
from streamlit import session_state as ss
from llm.Agent import Agent

from Chatting import client

rag_agent = Agent(tools=[])
st.title("RAG Agent")
st.sidebar.markdown(
    """
    # RAG Agent
    This is a agent that can retrieval documents and answer questions.
    He have a tools: retrieve documents from a vector store.
    """
)

import asyncio
asyncio.run(client(rag_agent))