from langchain_core.tools import tool

from src.tools.web_search import web_search
from src.tools.calculator import calculator


@tool
def rag_retrieve(query: str) -> str:
    """Search the internal knowledge base for documents relevant to the query.
    Use when the user asks about company docs, policies, or internal knowledge."""
    # Stub — schema exposed to Claude for tool selection.
    # Actual retrieval is handled in agent_loop.py via RAGRetriever (async).
    # Excluded from tool_map so it is never called directly.
    return ""


# TOOLS: list passed to llm.bind_tools() — Claude reads each tool's docstring
# and parameter schema to decide when and how to call each tool.
TOOLS: list = [web_search, calculator, rag_retrieve]

# tool_map: dict used in agent_loop.py to look up and execute a tool by name.
# rag_retrieve is excluded because the loop handles it via RAGRetriever directly.
tool_map: dict = {t.name: t for t in TOOLS if t.name != "rag_retrieve"}
