import streamlit as st
from streamlit import session_state as ss
from llm.Agent import Agent
from llm.Tools import get_time_tool, get_weather_tool


async def client(agent: Agent):
    for item in ss.messages:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    if not ss.loading:
        if new_prompt := st.chat_input("Send a message...", disabled=ss.loading):
            ss.prompt = new_prompt
            ss.messages.append({"role": "user", "content": new_prompt})
            ss.loading = True
            with st.chat_message("user"):
                st.write(new_prompt)
            st.rerun()

    if ss.loading:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, intermediate_steps = agent.chat(ss.prompt)
                if intermediate_steps:
                    st.write("Intermediate Steps:")
                    for step in intermediate_steps:
                        st.write(step)
                ss.loading = False
                ss.messages.append({"role": "assistant", "content": response})
                ss.prompt = ""
                st.rerun()