from langchain_core.tools import tool

from src.tools.web_search import web_search
from src.tools.calculator import calculator


@tool
def rag_retrieve(query: str) -> str:
    """Search the internal knowledge base for documents relevant to the query.
    Use when the user asks about company docs, policies, or internal knowledge."""
    # Stub: actual retrieval is handled by RAGRetriever in agent_loop.py.
    # This function exists only so Claude knows the tool is available.
    return ""


TOOLS: list = [web_search, calculator, rag_retrieve]

# rag_retrieve is excluded because the loop handles it via RAGRetriever directly
tool_map: dict = {t.name: t for t in TOOLS if t.name != "rag_retrieve"}
