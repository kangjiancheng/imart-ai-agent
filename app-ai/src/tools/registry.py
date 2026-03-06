# src/tools/registry.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is the "menu" of tools that Claude can choose from.
# When you add a new tool, you register it here — that's the only change needed.
#
# HOW TOOLS WORK (the @tool decorator):
#
# A LangChain tool is just a normal Python function with:
#   1. The @tool decorator applied to it
#   2. A descriptive docstring (Claude reads this to decide WHEN to call it)
#   3. Type hints on all parameters (used to generate a JSON schema)
#
# When we call llm.bind_tools(TOOLS), LangChain:
#   - Reads each tool's name, docstring, and parameter types
#   - Formats them as JSON schemas the Anthropic API understands
#   - Tells Claude "these tools are available; call them by returning a tool_call block"
#
# WHY TWO DATA STRUCTURES (TOOLS list + tool_map dict)?
#
#   TOOLS = [web_search, calculator, rag_retrieve]
#     → A LIST, because llm.bind_tools() expects a list.
#
#   tool_map = {"web_search": web_search, "calculator": calculator}
#     → A DICT, because in agent_loop.py we need to look up a tool BY ITS NAME.
#     → rag_retrieve is excluded because the loop handles RAG via RAGRetriever directly.
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.tools import tool
# @tool = the decorator that turns a plain Python function into a LangChain tool object

from src.tools.web_search import web_search
from src.tools.calculator import calculator
# Import the actual tool functions defined in their own files


@tool
def rag_retrieve(query: str) -> str:
    """Search the internal knowledge base for documents relevant to the query.
    Use when the user asks about company docs, policies, or internal knowledge."""
    # WHY IS THIS A STUB (empty function)?
    #
    # Claude needs to KNOW this tool exists so it can choose to call it.
    # bind_tools() reads the docstring and generates a schema — it doesn't care
    # whether the function body does real work.
    #
    # The ACTUAL retrieval (Milvus vector search) happens in agent_loop.py
    # when it detects tool_name == "rag_retrieve" and calls RAGRetriever.retrieve().
    #
    # This stub is excluded from tool_map below so it never gets called directly.
    return ""


# ── TOOLS: the list passed to llm.bind_tools() ───────────────────────────────
TOOLS: list = [web_search, calculator, rag_retrieve]
# : list = type hint saying "this is a list" (without specifying element type)
# Order doesn't matter — Claude reads all of them and picks based on the docstrings.

# ── tool_map: used in agent_loop.py to execute the chosen tool ───────────────
tool_map: dict = {t.name: t for t in TOOLS if t.name != "rag_retrieve"}
# PYTHON CONCEPT — dict comprehension:
# {t.name: t for t in TOOLS if t.name != "rag_retrieve"}
# = "for each tool t in TOOLS, create a key-value pair {name: tool},
#    but SKIP rag_retrieve because the loop handles it separately"
#
# t.name = the @tool decorator sets this automatically from the function name.
#   web_search  → t.name = "web_search"
#   calculator  → t.name = "calculator"
#
# Result: {"web_search": <web_search tool>, "calculator": <calculator tool>}
#
# In agent_loop.py:
#   tool = tool_map.get("calculator")   → returns the calculator tool object
#   result = await tool.ainvoke(args)   → executes it
