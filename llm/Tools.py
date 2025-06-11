from langchain_core.tools import tool, create_retriever_tool
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import requests
from Indexing import get_vectorstore
from llm.utils import meta_docs

@tool
def get_time_tool(tool_input: dict = None) -> str:
    "Get the current time in the format YYYY-MM-DD HH:MM:SS."
    "No input is required for this tool."
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather_tool(tool_input: dict = None) -> str:
    "Get the current weather for a specific location."
    "No input is required for this tool, it returns the weather for Hanoi, Vietnam."
    params = {
        "latitude": 21.0392903680844, 
        "longitude": 105.84138139562297
    }
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params
    )
    data = str(response.json())
    return data


@tool
def search_tool(querys: list[str]) -> list:
    "Search the web using DuckDuckGo."
    "Input is a list of strings, each string is a query."
    search = DuckDuckGoSearchRun(output_format="list")
    results = []
    for query in querys:
        result = search.invoke(query)
        if result:
            results.extend(result)
    return results
   

@tool
def retrieval_tool(query: str, key: str = "VN-History") -> list[Document]:
    """
    Use this tool to retrieve documents from a vectorstore.
    Args:
        key (str): The name of the vectorstore to retrieve from.
        query (str): The query to retrieve documents for.
    Returns:
        list[Document]: A list of documents that match the query.
    """
    index: FAISS = get_vectorstore(key)
    if not index:
        return []
    retriever = index.as_retriever()
    docs = retriever.invoke(query)
    return docs