import os

from langchain_core.tools import BaseTool

from src.config.settings import settings


def build_web_search_tool() -> BaseTool:
    """Create and return a configured TavilySearch tool, or a stub if not configured."""
    if not settings.tavily_api_key:
        from langchain_core.tools import tool

        @tool
        def web_search(query: str) -> str:  # noqa: ARG001
            """Search the web for current information, news, or recent events.
            Use when the question needs data that may not be in the knowledge base."""
            return "Live web search is currently unavailable."

        return web_search

    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

    from langchain_tavily import TavilySearch

    tool = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )
    return tool


web_search = build_web_search_tool()
