# Agent Tools — Web Search, Time Awareness, and Tool Design

> **Scope:** A practical guide to building and integrating tools into the `app-ai` LangChain agent.
> Covers the web search tool (Tavily), how Claude selects tools, time-awareness in the system prompt,
> and the design patterns used in `src/tools/`.
>
> For the RAG retrieval tool specifically, see [agent-rag.md](agent-rag.md).
> For the full agent loop that executes tools, see [app-ai-with-langchain.md](app-ai-with-langchain.md).

---

## 1. What Is a Tool?

A **tool** is a Python function that Claude can choose to call during the ReAct loop when it needs
information or capability outside its training data.

```text
Without tools:                       With tools:

User: "What is the latest            User: "What is the latest
       news about Iran?"                    news about Iran?"
                                                  │
LLM: (guesses from training data,               ▼
      may be months out of date)       Claude decides: call web_search
                                                  │
                                                  ▼
                                       web_search("latest Iran news", time_range="week")
                                                  │
                                                  ▼
                                       Tavily API → fresh results from today
                                                  │
                                                  ▼
                                       Claude reads results → accurate answer
```

Claude does **not** call the Python function directly. Instead it:
1. Reads tool schemas from the system prompt (injected by `bind_tools()`)
2. Returns a structured `tool_call` JSON object naming which tool to call and with what arguments
3. The agent loop (`agent_loop.py`) reads the JSON and executes the actual Python function
4. The result is sent back to Claude as a `ToolMessage`
5. Claude reasons over the result and either calls another tool or writes the final answer

---

## 2. How Claude Knows What Tools Exist

```python
# src/agent/agent_loop.py

from src.tools.registry import TOOLS, tool_map

planner = llm.bind_tools(TOOLS)
# bind_tools() does TWO things:
#   1. Reads every tool's name, docstring, and type hints → builds a JSON schema
#   2. Sends that schema to Claude on every request so it knows what tools exist
#
# The JSON schema for web_search looks like:
#   {
#     "name": "web_search",
#     "description": "Search the web for current information...",
#     "parameters": {
#       "type": "object",
#       "properties": { "query": { "type": "string" } },
#       "required": ["query"]
#     }
#   }
```

Claude reads the **docstring** to decide *when* to call the tool, and the **type hints** to know
*what arguments to pass*. A better docstring = better tool selection.

---

## 3. Tool Registry

All tools are registered in one place so `agent_loop.py` doesn't need to know about individual tools.

```python
# src/tools/registry.py

from src.tools.web_search import web_search
from src.tools.calculator import calculator
from src.tools.rag_tool import rag_retrieve

# TOOLS = list of tool objects passed to bind_tools()
# Claude receives the schema of every tool in this list on every request.
TOOLS = [web_search, calculator, rag_retrieve]

# tool_map = dict for fast lookup by name string
# When Claude returns tool_call = {"name": "calculator", "args": {...}},
# the agent loop does: tool = tool_map["calculator"], then tool.ainvoke(args)
tool_map = {tool.name: tool for tool in TOOLS}
# tool.name is set automatically from the function name by the @tool decorator
```

---

## 4. The @tool Decorator

The `@tool` decorator transforms a plain Python function into a LangChain Tool object.

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use for arithmetic, percentages, unit conversions, and numeric calculations.
    """
    # ...implementation...
```

After decoration, `calculator` is no longer a plain function — it becomes a Tool object with:

| Attribute | Source | Value |
|---|---|---|
| `.name` | Function name | `"calculator"` |
| `.description` | Docstring | `"Evaluate a mathematical expression..."` |
| `.args_schema` | Type hints | `{"expression": {"type": "string"}}` |
| `.invoke()` / `.ainvoke()` | Auto-wrapped | Calls the original function |

---

## 5. Web Search Tool — Tavily

### Why Tavily?

| | Tavily | DuckDuckGo | Google (SerpAPI) |
|---|---|---|---|
| LangChain integration | Official (`langchain-tavily`) | `langchain-community` | `langchain-community` |
| API key required | Yes (free tier: 1,000/month) | No | Yes (paid) |
| Result quality | AI-optimized snippets | Basic | Full Google results |
| Async support | Native | No | No |
| Per-invocation params | Yes (topic, time_range, domains) | No | Limited |
| Best for | AI agents | Quick prototyping | Highest coverage |

Tavily is the only search tool **officially recommended by LangChain** for agents. It returns clean
text snippets without HTML noise, ads, or pagination — designed specifically for LLM consumption.

### Installation

```bash
pip install langchain-tavily
# Get a free API key at https://app.tavily.com/sign-in
# Add to .env: TAVILY_API_KEY=tvly-xxxx
```

### Two packages — which to use?

| Package | Class | Status | Use when |
|---|---|---|---|
| `langchain-tavily` | `TavilySearch` | Current (recommended) | New projects |
| `langchain-community` | `TavilySearchResults` | Legacy | Already in use |

This project uses `langchain-tavily` → `TavilySearch`.

### TavilySearch parameters

```python
from langchain_tavily import TavilySearch

tool = TavilySearch(
    max_results=5,          # Number of results returned (default 5)
    topic="general",        # "general" | "news" | "finance"
    search_depth="basic",   # "basic" = 1 API credit, "advanced" = 2 API credits
    include_answer=False,   # Tavily's own AI-generated answer (CANNOT change at invocation time)
    include_raw_content=False,  # Full HTML content (CANNOT change at invocation time)
    include_images=False,   # Image URLs in results (can change at invocation time)
)
```

**Parameters that CAN be overridden per invocation (by Claude dynamically):**

| Parameter | Type | Example |
|---|---|---|
| `query` | str | `"Iran news March 2026"` |
| `topic` | str | `"news"` |
| `time_range` | str | `"day"` \| `"week"` \| `"month"` \| `"year"` |
| `search_depth` | str | `"advanced"` |
| `include_domains` | list | `["bbc.com", "reuters.com"]` |
| `exclude_domains` | list | `["reddit.com"]` |

**Parameters that CANNOT be overridden at invocation time** (would cause unpredictable context window
size): `include_answer`, `include_raw_content`.

### How Claude calls the tool dynamically

Because `time_range` and `include_domains` are invocation-time parameters, Claude can set them
per query without any code change. Example tool call Claude might generate:

```json
{
  "name": "web_search",
  "args": {
    "query": "latest Iran news",
    "time_range": "week",
    "include_domains": ["reuters.com", "bbc.com"]
  }
}
```

This is more powerful than a fixed `@tool` function — the agent adapts the search strategy per query.

### Factory pattern — why not instantiate TavilySearch directly?

```python
# src/tools/web_search.py

def build_web_search_tool() -> BaseTool:
    """
    Factory function: creates TavilySearch if the API key is configured,
    or returns a safe stub @tool if it is not.
    """
    if not settings.tavily_api_key:
        @tool
        def web_search(query: str) -> str:  # noqa: ARG001
            """Search the web for current information, news, or recent events.
            Use when the question needs data that may not be in the knowledge base."""
            return "Live web search is currently unavailable."
        return web_search

    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    from langchain_tavily import TavilySearch
    return TavilySearch(max_results=5, topic="general", search_depth="basic")

web_search = build_web_search_tool()
```

**Why a factory?**

1. Deferred import — if `langchain-tavily` is not installed, the app still starts; only web search
   is unavailable
2. Safe fallback — missing API key returns a stub tool with the same name and schema, so
   `registry.py` never needs to change
3. API key injection — `TavilySearch` reads `TAVILY_API_KEY` from `os.environ` at instantiation
   time; we set it just before construction

---

## 6. Time Awareness — Why Claude Returns Stale Results

### The problem

Claude is a **stateless language model with a knowledge cutoff**. Without being told the current
date, it has no idea what "latest" or "recent" means. When it calls the web search tool without
`time_range`, Tavily may return cached results from months ago.

**Real example from this project:**

```
User: "Could you list some of the latest news about Iran?"
Agent: (no date in system prompt → no time_range in search call)
Result: News from June–July 2025 presented as "latest" — 9 months out of date.
```

### The fix — inject the current date into the system prompt

```python
# src/llm/claude_client.py

from datetime import datetime

def build_system_prompt(...) -> str:
    return f"""You are a helpful AI assistant with access to tools and a knowledge base.
You are running on model: {model}
Current date and time: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Rules
- When searching for news or recent events, always pass `time_range="week"` or
  `time_range="month"` to the web_search tool so results are filtered to the
  relevant period — never rely on the default which may return outdated content.
...
"""
```

Now Claude knows it is March 12, 2026, and understands that "latest" means the past few days —
not its training cutoff date.

### Server time vs UTC vs client time

| Option | How | Use when |
|---|---|---|
| `datetime.now()` | Server local time | Server and users are in the same region |
| `datetime.now(timezone.utc)` | UTC, timezone-unambiguous | Server deployed across regions |
| Client-provided date | Passed in request body | User's local date differs from server's |

This project uses `datetime.now()` (server local time). For a deployed multi-region service,
switch to UTC:

```python
from datetime import datetime, timezone
datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
```

The `user_context.timezone` field in the request schema can also be injected into the system prompt
if users are in very different timezones from the server.

### Why Claude can't "just know" the date

Claude does not have access to a clock. Every request it receives is stateless — it only sees:
1. The system prompt
2. The conversation history passed in the request

There is no background process, no timer, no connection to the internet. The system prompt
*is* the mechanism for passing runtime context like the current date.

---

## 7. Tool Design Patterns

### Pattern 1 — Keep tools thin, return strings

Tools should do one thing and return a plain string. Claude handles all reasoning.

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for any arithmetic or numeric calculation."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
```

### Pattern 2 — Return errors as strings, never raise

A tool that raises an exception crashes the ReAct loop. Return the error as a string — Claude
receives it as a ToolMessage observation and decides how to recover.

```python
try:
    result = call_external_api(args)
    return result
except Exception as e:
    return f"Tool failed: {str(e)}"
    # Claude sees this as an observation and may retry with different args,
    # use a different tool, or answer from its own knowledge.
```

### Pattern 3 — Write docstrings for Claude, not for humans

The docstring is the primary signal Claude uses to decide *when* to call the tool. Write it as
instructions to Claude.

```python
# BAD — vague, Claude may not understand when to use this
@tool
def web_search(query: str) -> str:
    """Search the web."""

# GOOD — tells Claude exactly when to call this tool
@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or recent events.
    Use when the question needs data that may not be in the knowledge base.
    Do NOT use for math, definitions, or questions answerable from training data."""
```

### Pattern 4 — Async tools with ainvoke

All tools in this project are called with `tool.ainvoke(args)` in the agent loop. The `@tool`
decorator wraps synchronous functions automatically so `ainvoke()` works on all of them.

For tools that make async calls (database, HTTP), define the underlying function as `async`:

```python
from langchain_core.tools import tool

@tool
async def fetch_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a ticker symbol. Use for finance questions."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/price/{ticker}") as resp:
            data = await resp.json()
            return f"{ticker}: ${data['price']}"
```

### Pattern 5 — Stub tools for optional dependencies

If a tool depends on an optional service (like Tavily), return a stub that keeps the same name
and schema. This prevents registry.py or agent_loop.py from needing conditional logic.

```python
def build_optional_tool() -> BaseTool:
    if not settings.optional_api_key:
        @tool
        def my_tool(query: str) -> str:  # noqa: ARG001
            """..."""
            return "This feature is currently unavailable."
        return my_tool

    # real implementation
    return RealTool(api_key=settings.optional_api_key)
```

---

## 8. Available Search Libraries Comparison

| Library | Package | LangChain class | API key | Notes |
|---|---|---|---|---|
| **Tavily** | `langchain-tavily` | `TavilySearch` | Required | Best for AI agents, official LangChain pick |
| **Brave Search** | `langchain-community` | `BraveSearch` | Required | Privacy-focused, affordable |
| **SerpAPI** | `langchain-community` | `SerpAPIWrapper` | Required | Real Google results, pricier |
| **You.com** | `langchain-community` | `YouSearchAPIWrapper` | Required | Good RAG-style snippets |
| **DuckDuckGo** | `duckduckgo-search` | `DuckDuckGoSearchRun` | None | Free, rate-limited, no async |
| **Jina Reader** | `langchain-community` | `JinaSearch` | Optional | Converts URLs to LLM-readable markdown |

For this project (LangChain + Claude, news/current events): **Tavily** is the right choice.

---

## 9. Tool Execution Flow in agent_loop.py

```text
① planner = llm.bind_tools(TOOLS)
   └── Claude receives JSON schema for every tool

② response = await planner.ainvoke(messages)
   └── Claude returns AIMessage with .tool_calls list:
       [{"name": "web_search", "args": {"query": "...", "time_range": "week"}, "id": "tc_001"}]

③ for tool_call in response.tool_calls:
       tool = tool_map[tool_call["name"]]         # look up tool by name
       result = await tool.ainvoke(tool_call["args"])  # execute tool

④ messages.append(response)                       # AIMessage with tool_calls
   messages.append(ToolMessage(                   # tool result
       content=str(result),
       tool_call_id=tool_call["id"]               # id must match — Anthropic validates 1:1
   ))

⑤ Loop back to ② — Claude sees the tool result and decides next step
```

**Critical:** Every `tool_use` block in the AIMessage must be followed by a matching
`tool_result` ToolMessage with the same `id`. Missing a match causes an Anthropic API 400 error.

---

## 10. Files

```
app-ai/
└── src/
    ├── tools/
    │   ├── registry.py       TOOLS list + tool_map dict — register tools here
    │   ├── web_search.py     TavilySearch factory — web search tool
    │   ├── calculator.py     @tool — safe expression evaluator
    │   └── rag_tool.py       @tool stub — RAG retrieve (async, handled specially in agent_loop)
    └── llm/
        └── claude_client.py  build_system_prompt() — injects datetime.now() into every request
```

Cross-references:
- [app-ai-with-langchain.md](app-ai-with-langchain.md) — the ReAct loop that calls tools
- [agent-rag.md](agent-rag.md) — RAG retrieval tool in depth (Milvus, embeddings, chunking)
- [app-ai-agent-memory.md](app-ai-agent-memory.md) — long-term memory (also runs as a tool result)
