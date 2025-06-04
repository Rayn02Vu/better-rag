import streamlit as st
from streamlit import session_state as ss

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



api_key = st.secrets.get("CONO_API_KEY", None)

class Agent:
    def __init__(self, tools: list = None, system_prompt: str = None):
        self.llm = ChatOpenAI(
            model="cono-3-exp",
            api_key=api_key,
            base_url="https://api.arcanic.ai/v1",
            temperature=0.5,
        )
        self.system_prompt = system_prompt if system_prompt else (
            "You are a helpful assistant. "
            "You can answer questions, provide information, and assist with tasks. "
            "Use the tools provided to help you answer the user's questions."
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.tools = tools if tools else []
        self.executor = None
        self.build()
    
    def build(self):
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools,
        )
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def chat(self, query: str):
        if not query:
            pass
        try:
            response = self.executor.invoke(
                {"input": query, "chat_history": ss.messages, "agent_scratchpad": []}
            )
            return response.get("output", "No response from agent."), response.get("intermediate_steps", [])
        except Exception as e:
            print(f"Error: {e}")
            st.error(f"An error occurred: {e}")
            return "Error processing the request.", []
        
