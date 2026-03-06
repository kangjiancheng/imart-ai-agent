# src/tools/web_search.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS TOOL?
#
# This file defines the `web_search` tool — one of the tools Claude can call
# during the ReAct loop to answer questions that need current or recent information.
#
# WHAT IS A "TOOL" IN LANGCHAIN?
#   A tool is a Python function that Claude can choose to call when it decides
#   that information from outside its training data is needed.
#   Claude does NOT call the function directly — it returns a JSON "tool_call"
#   object describing which tool to call and what arguments to pass.
#   The agent loop reads that JSON and calls the actual Python function here.
#
# HOW DOES CLAUDE KNOW THIS TOOL EXISTS?
#   In registry.py, `TOOLS = [web_search, calculator, rag_retrieve]`.
#   The LLM is initialized with `llm.bind_tools(TOOLS)`.
#   LangChain reads the @tool decorator's docstring and type hints to build a
#   JSON schema describing the tool. That schema is sent to Claude on every
#   request so Claude knows what tools are available.
#
# WHY TAVILY (not Google or Bing)?
#   Tavily is an AI-optimized search API that returns clean text snippets
#   without HTML noise, ads, or pagination — perfect for LLM consumption.
#   Requires a free TAVILY_API_KEY from tavily.com.
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.tools import tool
# @tool = a decorator from LangChain that transforms a plain Python function
# into a LangChain Tool object. It auto-reads:
#   - The function name → tool name Claude uses to call it ("web_search")
#   - The docstring → tool description Claude reads to decide WHEN to use it
#   - The type hints (query: str) → JSON schema Claude uses to structure its call

from src.config.settings import settings
# settings = the singleton Settings instance from config/settings.py
# Used here to read settings.tavily_api_key without re-reading .env


# ── @tool decorator ───────────────────────────────────────────────────────────
# PYTHON CONCEPT — decorators:
#   A decorator is a function that WRAPS another function to add behavior.
#   Syntax: @decorator_name above a function definition.
#   `@tool` here transforms `web_search` from a plain function into a
#   LangChain Tool object that has .name, .description, .args_schema, and .invoke().
#   In TypeScript: similar to a class decorator or a HOF wrapper.
#
# WHAT THE DOCSTRING DOES:
#   The docstring (the string between """) is NOT just documentation for humans.
#   LangChain's @tool reads it and sends it to Claude as the tool description.
#   Claude reads: "Search the web for current information, news, or recent events.
#   Use when the question needs data that may not be in the knowledge base."
#   and uses that to decide WHEN to call web_search vs rag_retrieve vs calculator.
#   A better docstring = better tool selection by Claude.

@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or recent events.
    Use when the question needs data that may not be in the knowledge base."""

    # ── Guard: check API key before importing the library ──────────────────
    # PYTHON CONCEPT — truthiness check on string:
    #   `if not settings.tavily_api_key` is True when:
    #     - tavily_api_key is None (not set in .env)
    #     - tavily_api_key is "" (empty string, our default in settings.py)
    #   Both None and "" are "falsy" in Python (like null/undefined/"" in JS).
    #   This guard prevents crashing if the user hasn't configured Tavily.
    #
    # EFFECT ON THE AGENT:
    #   Returns an error string instead of raising an exception.
    #   Claude receives the error string as the tool result and can tell the user
    #   "Web search is not configured" rather than the agent crashing with a 500.
    if not settings.tavily_api_key:
        return "Web search is not configured (TAVILY_API_KEY missing)."

    # ── Try/except: graceful failure ──────────────────────────────────────
    # PYTHON CONCEPT — try/except:
    #   Same as try/catch in JavaScript.
    #   Any exception raised inside `try` is caught by `except`.
    #   `Exception as e` binds the exception object to variable `e`.
    #
    # WHY RETURN A STRING INSTEAD OF RAISING?
    #   This is a deliberate design choice: tool errors should NEVER crash the loop.
    #   If web search fails (network timeout, API quota exceeded, etc.),
    #   Claude receives the error string as the tool result.
    #   Claude can then decide to try another tool or answer from its own knowledge.
    #   Raising an exception would terminate the entire agent loop and return 500.
    try:
        # PYTHON CONCEPT — deferred / local import:
        #   `from tavily import TavilyClient` is inside the function body, not at
        #   the top of the file. This is called a "deferred import" or "local import".
        #   Why? If tavily is not installed, importing at module load time would crash
        #   the entire app on startup. Importing here means the error only triggers
        #   when this function is actually called, and the try/except catches it.
        #   In production, tavily should be in requirements.txt and always available.
        from tavily import TavilyClient

        # Create a Tavily client with our API key
        client = TavilyClient(api_key=settings.tavily_api_key)

        # Perform the search — max_results=3 keeps the context window small
        # results is a dict: { "results": [ { "content": "...", "url": "..." }, ... ] }
        results = client.search(query=query, max_results=3)

        # ── List comprehension to extract text snippets ──────────────────
        # PYTHON CONCEPT — list comprehension:
        #   [expression for item in iterable]
        #   = build a new list by applying `expression` to each `item`.
        #
        #   `[r.get("content", "") for r in results.get("results", [])]`
        #   Breakdown:
        #     results.get("results", [])
        #       = read the "results" key from the dict; if missing, use []
        #       .get(key, default) is safer than results["results"] because it
        #       won't crash if the key doesn't exist.
        #     for r in ...
        #       = iterate over each result dict in the list
        #     r.get("content", "")
        #       = read the "content" field from this result; if missing, use ""
        #   Result: a list of text strings, one per search result.
        #   In TypeScript: results?.results?.map(r => r.content ?? "") ?? []
        snippets = [r.get("content", "") for r in results.get("results", [])]

        # ── Join snippets and handle empty case ──────────────────────────
        # "\n\n".join(snippets)
        #   = concatenate all snippets with a blank line between each.
        #   In TypeScript: snippets.join("\n\n")
        #
        # `or "No results found."` — Python boolean short-circuit:
        #   If join() returns an empty string "" (falsy), Python evaluates the
        #   right side and returns "No results found." instead.
        #   In TypeScript: snippets.join("\n\n") || "No results found."
        return "\n\n".join(snippets) or "No results found."

    except Exception as e:
        # f-string = Python's template literal syntax (same as JS template literals).
        # f"Web search failed: {str(e)}"
        #   str(e) converts the exception object to a readable error message string.
        # Claude receives this string as the tool result and handles it gracefully.
        return f"Web search failed: {str(e)}"
