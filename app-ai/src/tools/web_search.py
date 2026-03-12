# src/tools/web_search.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS TOOL?
#
# This file defines the `web_search` tool — one of the tools Claude can call
# during the ReAct loop to answer questions that need current or recent information.
#
# WHAT IS A "TOOL" IN LANGCHAIN?
#   A tool is a Python function (or object) that Claude can choose to call when
#   it decides that information from outside its training data is needed.
#   Claude does NOT call the function directly — it returns a JSON "tool_call"
#   object describing which tool to call and what arguments to pass.
#   The agent loop reads that JSON and calls the actual Python function here.
#
# HOW DOES CLAUDE KNOW THIS TOOL EXISTS?
#   In registry.py, `TOOLS = [web_search, calculator, rag_retrieve]`.
#   The LLM is initialized with `llm.bind_tools(TOOLS)`.
#   LangChain reads each tool's name, description, and args schema to build a
#   JSON schema that is sent to Claude on every request.
#
# WHY TAVILY (not Google or Bing)?
#   Tavily is an AI-optimized search API that returns clean text snippets
#   without HTML noise, ads, or pagination — perfect for LLM consumption.
#   It is also the officially recommended search tool by LangChain.
#   Requires a free TAVILY_API_KEY from tavily.com (1,000 free searches/month).
#
# WHY langchain-tavily INSTEAD OF THE RAW tavily PACKAGE?
#   The `langchain-tavily` package provides a `TavilySearch` class that is
#   a proper LangChain BaseTool subclass. This means:
#     - It supports native async (no extra wrapper needed)
#     - The agent can dynamically set parameters per invocation
#       (e.g., include_domains, time_range, search_depth)
#     - It returns structured ToolMessage output, not raw JSON strings
#     - It integrates with LangChain's tool-call flow without any glue code
# ─────────────────────────────────────────────────────────────────────────────

import os
# os.environ lets us inject the API key before TavilySearch reads it.
# TavilySearch reads TAVILY_API_KEY from the environment at instantiation time.

from langchain_core.tools import BaseTool
# BaseTool is the abstract base class all LangChain tools inherit from.
# We use it here only for the type hint on the factory function's return type.

from src.config.settings import settings
# settings = the singleton Settings instance from config/settings.py
# Used here to read settings.tavily_api_key without re-reading .env directly.


def build_web_search_tool() -> BaseTool:
    """
    Factory function that creates and returns a configured TavilySearch tool.

    WHY A FACTORY FUNCTION?
      TavilySearch reads TAVILY_API_KEY from os.environ at instantiation time.
      By wrapping it in a function, we can:
        1. Check if the key is configured before trying to import langchain-tavily.
        2. Set os.environ["TAVILY_API_KEY"] right before instantiation.
        3. Return a fallback @tool stub if Tavily is unavailable, so the rest
           of the app (registry.py, agent loop) doesn't need to change.

    WHAT IS A "FACTORY FUNCTION"?
      A factory function is a function whose job is to CREATE and return an object.
      Instead of calling MyClass() directly, you call make_my_class() and let it
      handle any setup logic before construction.
      In TypeScript: a function that returns `new MyClass(...)` after some guards.
    """

    # ── Guard: check API key before trying to import the library ──────────────
    # `if not settings.tavily_api_key` is True when:
    #   - tavily_api_key is None (not set in .env)
    #   - tavily_api_key is "" (empty string, the default in settings.py)
    # Both None and "" are "falsy" in Python (like null/undefined/"" in JS).
    # This guard prevents a confusing error from langchain-tavily about a
    # missing API key and instead returns a safe fallback tool.
    if not settings.tavily_api_key:
        # ── Fallback: return a stub @tool if Tavily is not configured ─────────
        # PYTHON CONCEPT — importing inside a branch:
        #   We only import `tool` here because this branch is rarely taken.
        #   The main path (below) uses TavilySearch directly without @tool.
        from langchain_core.tools import tool

        @tool
        def web_search(query: str) -> str:  # noqa: ARG001
            """Search the web for current information, news, or recent events.
            Use when the question needs data that may not be in the knowledge base."""
            # `query` is intentionally unused — this is a stub that fires when
            # TAVILY_API_KEY is not configured. The parameter must stay in the
            # signature so LangChain builds the correct JSON schema for Claude.
            # Return a generic message — do NOT reveal internal config details
            # like env var names or library names. Claude will rephrase this
            # as a user-friendly message.
            return "Live web search is currently unavailable."

        return web_search

    # ── Inject API key into environment for TavilySearch ──────────────────────
    # TavilySearch (and most LangChain community tools) read credentials from
    # os.environ at __init__ time. Setting it here ensures Tavily always gets
    # the key from our settings singleton, which already read from .env.
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

    # ── Import and instantiate TavilySearch ───────────────────────────────────
    # PYTHON CONCEPT — deferred / local import:
    #   The import is inside the function body rather than at the top of the file.
    #   Why? If `langchain-tavily` is not installed, a top-level import would crash
    #   the entire app at startup. A local import only fails when this function is
    #   called, and only affects this tool — not the whole app.
    #   In production, langchain-tavily should be in requirements.txt.
    from langchain_tavily import TavilySearch

    # TavilySearch is a LangChain BaseTool subclass (not a raw function).
    # Parameters set here are the DEFAULTS for every invocation.
    # The agent can still OVERRIDE: include_domains, exclude_domains,
    # time_range, search_depth, and include_images per query.
    # Parameters that CANNOT be overridden at invocation time (they affect
    # response size and could cause context window issues):
    #   - include_answer
    #   - include_raw_content
    tool = TavilySearch(
        max_results=5,          # Return up to 5 search results per query
        topic="general",        # "general" | "news" | "finance"
        search_depth="basic",   # "basic" = 1 API credit, "advanced" = 2 credits
        include_answer=False,   # Don't include Tavily's own AI-generated answer
        include_raw_content=False,  # Don't include full HTML — keeps context small
        include_images=False,   # No images — this is a text-only agent
    )

    return tool


# ── Module-level tool instance ────────────────────────────────────────────────
# PYTHON CONCEPT — module-level execution:
#   Code at the top level of a module runs ONCE when the module is first imported.
#   `web_search` is created here so registry.py can import it with:
#       from src.tools.web_search import web_search
#   The factory handles all guards and fallbacks, so registry.py stays clean.
web_search = build_web_search_tool()
