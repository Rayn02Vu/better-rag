import streamlit as st
from streamlit import session_state as ss
from llm.Agent import Agent
from llm.Tools import get_time_tool, get_weather_tool
from Chatting import client

simple_agent = Agent(tools=[get_time_tool, get_weather_tool])
st.title("Simple Agent")
st.sidebar.markdown(
    """
    # Simple Agent
    This is a simple agent that can answer questions.
    He have some tools: get current time, get weather, and get news.
    """
)

import asyncio
asyncio.run(client(simple_agent))