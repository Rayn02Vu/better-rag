from numpy import isin
import streamlit as st
from streamlit import session_state as ss

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm.LLM import LLM

api_key = st.secrets.get("CONO_API_KEY", None)

class Agent:
    def __init__(
            self, 
            tools: list = None, 
            system_prompt: str = None,
            prompt_template: ChatPromptTemplate = None
        ):
        self.llm = LLM()

        self.system_prompt = system_prompt or self.llm.system_prompt

        self.prompt = prompt_template or ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Có thể tham khảo thêm các context sau nếu cảm thấy cần thiết: {context}"),
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
    
    def chat(self, query: str, context: str | list[str] = None) -> str:
        if not query:
            pass
        if isinstance(context, list):
            context = "\n".join(context)
        try:
            response = self.executor.invoke(
                {
                    "input": query, 
                    "chat_history": ss.messages, 
                    "context": context if context else "", 
                    "agent_scratchpad": []
                }
            )
            return response.get("output", "No response from agent.")
        except Exception as e:
            print(f"Error: {e}")
            st.error(f"An error occurred: {e}")
            return "Error processing the request."
        
