from langchain_core.tools import tool, create_retriever_tool
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
import requests


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
   
