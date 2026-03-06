# app-ai — Phase 1: LangChain Agent (Manual Loop)

> **Scope:** This document is a complete guide to building the `app-ai` agent using **LangChain only** — no LangGraph required.
> Build this first. The agent is fully functional at this point.
>
> **Next phase:** Once this is working, see [app-ai-with-langgraph.md](app-ai-with-langgraph.md) to upgrade the loop internals to a LangGraph `StateGraph`.
> For the full system context (Next.js → NestJS → FastAPI → Claude), see [app-architecture.md](app-architecture.md).

---

## 1. What You Will Build

A FastAPI service (`app-ai`) that receives a user message from NestJS and runs an AI agent loop using Claude as the reasoning engine. The agent can call tools (web search, calculator, code execution, database queries) and retrieve documents from a vector store (RAG) before streaming a final answer back to the browser.

```text
Client (Next.js / React)
        │  HTTPS / WebSocket
        ▼
API Gateway (Nginx / Cloudflare)
        │
        ▼
app-service (NestJS — BFF)
        │  HTTP internal
        │  sends: { message, history, userId, userContext }
        ▼
app-ai (FastAPI — Python)         ← this document
        │
        ▼
AI Agent Runtime — LangChain stack
        │
        ├── ChatAnthropic via langchain-anthropic  (LLM reasoning + tool calling)
        ├── Manual for loop in agent_loop.py       (agent iteration control)
        ├── Milvus via pymilvus                    (vector search — RAG + long-term memory)
        └── LangChain @tool functions              (web search, code exec, calculator, SQL)
```

`app-ai` is **stateless**. It receives full context on every request from `app-service` and never stores user state — that responsibility belongs to NestJS and PostgreSQL.

---

## 2. LangChain Components to Learn First

These are the building blocks used throughout this guide. Learn them before reading further.

```text
ChatAnthropic          LLM client from langchain-anthropic; wraps the Anthropic API
bind_tools(TOOLS)      attach tool schemas to an LLM call; Claude returns structured tool_calls
@tool decorator        wrap a plain Python function as a LangChain-compatible tool
HumanMessage           typed user message  (replaces {"role": "user", "content": ...})
AIMessage              typed assistant message (carries .tool_calls when Claude selects a tool)
SystemMessage          typed system prompt
ToolMessage            typed tool result fed back to the LLM after a tool executes
llm.astream()          stream response tokens one-by-one from the LLM
llm.ainvoke()          async call — waits for the full response (used in the planner)
VectorMemory           long-term user memory stored in Milvus
EmbeddingClient        convert text to vectors for RAG search
```

---

## 3. Context Payload from NestJS

Every request arriving at `app-ai` carries a normalized payload assembled by `app-service`:

```json
{
  "user_id": "usr_abc123",
  "message": "What are the benefits of RAG?",
  "history": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help?" }
  ],
  "user_context": {
    "subscription_tier": "pro",
    "locale": "en-US",
    "timezone": "America/New_York"
  },
  "session_id": "sess_xyz789"
}
```

The Input Handler (Component 1) normalizes this payload before any agent logic runs. The agent never reads raw HTTP headers or JWT tokens — that boundary is enforced at the NestJS layer.

---

## 4. Core Architecture — ReAct Loop

The agent follows the **ReAct pattern**: Reason → Act → Observe → repeat. With LangChain, this is a plain Python `for` loop — every step is explicit code you can read line by line.

```text
NestJS Request
      │
      ▼
input_handler()   ← parse + validate payload (Pydantic)
      │
      ▼
guardrails()      ← content policy · PII · injection check
      │ fail ──► 422 Error
      ▼
memory_recall()   ← load long-term context from Milvus
      │
      ▼
build_messages()  ← assemble system prompt + history + user message
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  planner = llm.bind_tools(TOOLS)                     │
│                                                      │
│  for i in range(MAX_ITERATIONS):                     │
│                                                      │
│    response = await planner.ainvoke(messages)        │
│                                                      │
│    if no tool_calls ──► break  (final answer)        │
│                                                      │
│    elif tool == "rag_retrieve" → retriever.query()   │
│    else                        → tool.ainvoke()      │
│                                                      │
│    messages.append(response)       ← assistant turn  │
│    messages.append(ToolMessage())  ← observation     │
│                                                      │
└──────────────────────────────────────────────────────┘
      │
      ▼
async for chunk in llm.astream(messages):
    yield chunk.content    ← SSE tokens to browser
      │
      ▼
memory_write()   ← summarize + store to Milvus
```

**How the loop controls itself:**

- `bind_tools(TOOLS)` tells Claude which tools exist. Claude responds with a structured `tool_calls` list instead of plain text — no JSON parsing needed.
- If `response.tool_calls` is empty, Claude is done reasoning. Stream the final answer and `break`.
- If `response.tool_calls` has an entry, execute the tool, append a `ToolMessage` with the result, and loop again.
- The `messages` list grows each iteration — this is the agent's working memory for this request.

---

## 5. The Seven Core Components

---

### Component 1 — Input Handler

Prepares and validates the incoming request before any agent logic runs.

**Responsibilities:**

```text
Receive NestJS payload
Parse and validate with Pydantic
Attach conversation history
Identify user context (tier, locale)
Normalize message encoding
Estimate incoming token count
Reject malformed requests early (400)
```

**Pydantic model:**

```python
# app/schemas/request.py
#
# Pydantic is a data-validation library. When the NestJS request arrives as JSON,
# Pydantic automatically parses it into these Python objects AND enforces all the
# constraints (min_length, required fields, allowed values). If validation fails,
# FastAPI returns a 422 error automatically — no manual if-checks needed.

from pydantic import BaseModel, Field
from typing import Literal

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]  # only these two strings are allowed
    content: str

class UserContext(BaseModel):
    subscription_tier: str
    locale: str = "en-US"    # default: if NestJS omits this, Python uses "en-US"
    timezone: str = "UTC"

class AgentRequest(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1, max_length=10_000)
    #               ^^^  Field(...) means "required"
    #               min_length=1 → empty strings rejected
    #               max_length=10_000 → protects against huge payloads
    history: list[HistoryMessage] = []   # empty list for first message
    user_context: UserContext
    session_id: str
```

---

### Component 2 — Guardrails

A safety layer that runs **before the LLM** receives any input. It cannot be bypassed.

**Three checks, in order:**

```text
1. Content policy
   └── Reject harmful, illegal, or policy-violating requests
       Return 422 with reason code

2. PII detection
   └── Detect and redact: email, phone, SSN, credit card patterns
       Log redaction event for compliance audit

3. Prompt injection detection
   └── Detect instruction-override patterns in user message
       e.g. "Ignore previous instructions and..."
       Sanitize or reject
```

**Implementation:**

```python
# app/guardrails/checker.py

class GuardrailChecker:
    def check(self, message: str) -> GuardrailResult:
        result = GuardrailResult(passed=True, sanitized_message=message)

        # CHECK 1: content policy — early exit on failure (saves CPU)
        result = self._check_content_policy(result)
        if not result.passed:
            return result

        # CHECK 2: PII redaction — scrubs data in-place, does NOT fail the request
        result = self._redact_pii(result)

        # CHECK 3: prompt injection — mutates sanitized_message or flips passed
        result = self._check_injection(result)

        return result   # caller reads result.passed and result.sanitized_message
```

Guardrails run synchronously and must complete in under 100ms. They never call the LLM.

---

### Component 3 — Planner

The planner is the core of the agent loop. On each iteration it calls the LLM with `bind_tools()` and interprets the response to decide what to do next.

**ReAct pattern (Thought → Action → Observation):**

```text
User question: "What are the benefits of RAG?"

Iteration 1:
  planner calls llm.bind_tools(TOOLS).ainvoke(messages)
  Claude responds with tool_call → rag_retrieve("benefits of RAG systems")
  → execute rag_retrieve, append ToolMessage(result)
  → loop continues

Iteration 2:
  planner calls llm.bind_tools(TOOLS).ainvoke(messages)
  Claude responds with NO tool_call → this is the final answer
  → break loop, stream response
```

**How tool-based routing works:**

```python
# Inside agent_loop.py

response = await planner.ainvoke(messages)
# response.tool_calls is a structured list, e.g.:
#   [{"name": "rag_retrieve", "args": {"query": "benefits of RAG"}, "id": "tc_001"}]

if not response.tool_calls:
    # No tool call = Claude is ready to answer.
    # Stream the final answer and stop the loop.
    async for chunk in llm.astream(messages):
        yield chunk.content
    break

# Tool call present — identify which tool and execute it.
tool_call = response.tool_calls[0]
tool_name = tool_call["name"]   # e.g. "rag_retrieve"
tool_args = tool_call["args"]   # e.g. {"query": "benefits of RAG"}
```

---

### Component 4 — Tool Registry and Executor

Tools extend what the agent can do beyond LLM reasoning. With LangChain, tools are defined using the **`@tool` decorator** — the docstring becomes the tool description that Claude reads to decide when to call it.

**Defining tools:**

```python
# app/tools/web_search.py
#
# The @tool decorator does three things:
#   1. Wraps the function as a LangChain-compatible tool object
#   2. Uses the docstring as the tool description injected into the LLM prompt
#      (Claude reads this to decide when to call the tool)
#   3. Generates a JSON schema from the type hints so bind_tools()
#      can tell the Anthropic API exactly what arguments this tool accepts

from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use when the question needs recent
    data, news, or facts that may not be in the knowledge base."""
    # function body calls Tavily / SerpAPI here
    ...

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Use for arithmetic, percentages,
    unit conversions. Input must be a valid math expression like '15 * 1.08'."""
    ...

@tool
def code_exec(code: str) -> str:
    """Execute Python code in a sandboxed subprocess and return stdout.
    Use for data analysis, formatting, or computation that needs real code."""
    ...

@tool
def database_query(sql: str) -> str:
    """Run a READ-ONLY SQL query against the internal database.
    Never accepts INSERT, UPDATE, DELETE, or DROP statements."""
    ...
```

**Tool registry:**

```python
# app/tools/registry.py
#
# The registry is a LIST — llm.bind_tools(TOOLS) expects a list of LangChain tool objects.
# LangChain reads each tool's .name and .description automatically.

from app.tools.web_search import web_search
from app.tools.calculator  import calculator
from app.tools.code_exec   import code_exec
from app.tools.database    import database_query

TOOLS: list = [web_search, calculator, code_exec, database_query]

# For the executor, build a name→tool lookup from the same list:
tool_map: dict = {t.name: t for t in TOOLS}
# The @tool decorator sets .name from the function name automatically.
```

**Executing a tool in the loop:**

```python
# Inside agent_loop.py

tool_name = tool_call["name"]   # e.g. "web_search"
tool_args = tool_call["args"]   # e.g. {"query": "Tesla revenue 2024"}

if tool_name == "rag_retrieve":
    # RAG is handled separately via the retriever
    result = await Retriever().query(tool_args["query"])
else:
    tool = tool_map.get(tool_name)
    if tool is None:
        result = f"Error: unknown tool '{tool_name}'"
    else:
        try:
            result = await tool.ainvoke(tool_args)
        except Exception as e:
            result = f"Error: {tool_name} failed — {str(e)}"
            # Tool errors become observations — the LLM sees this and adapts.
            # The loop never crashes due to a tool failure.

# Feed the result back to the LLM as a ToolMessage
messages.append(response)                                             # assistant turn
messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
# tool_call_id links this result to the tool_call that requested it.
# Without it, the Anthropic API returns a validation error.
```

---

### Component 5 — Memory System

> For the full guide (what memory is, why it matters, how each scope works, failure handling, and design rules), see [agent-memory.md](agent-memory.md).

The agent uses two memory scopes.

#### Short-term memory — local variables in `run()`

Scope: **one request**. Held in local Python variables inside `agent_loop.run()`. Destroyed when the function returns.

```python
# Inside run():
messages    = [...]       # the growing message list (short-term working memory)
iterations  = []          # what happened each loop: {"tool": ..., "result": ...}
tokens_used = 0           # running token count
```

This is the agent's "scratchpad" for this request — it accumulates the full reasoning chain (tool calls and observations) so the LLM sees all prior context on each iteration.

#### Long-term memory — `app/memory/vector_memory.py`

Scope: **across sessions**. Stored in Milvus (`user_memory` collection, per-user filtered). Persists indefinitely until explicitly deleted.

Holds user preferences, past task summaries, and domain context. Recalled **once per request before the loop starts** and injected into the system prompt. Written **after the response is streamed**, never during the loop.

```python
class VectorMemory:
    COLLECTION = "user_memory"
    MIN_SCORE  = 0.78    # higher threshold than RAG — recall must be confident

    async def recall(self, user_id: str, query: str, top_k: int = 5) -> list[MemoryChunk]: ...
    async def store_if_new(self, user_id: str, content: str, tags: list[str]) -> None: ...
    async def forget(self, user_id: str, memory_id: str) -> None: ...  # GDPR deletion
```

#### Memory position in the LLM prompt (order matters)

```text
1. System Prompt          ← agent identity, tool list, rules
2. Long-term Memory       ← recalled before loop; shapes all reasoning that follows
3. Conversation History   ← prior turns from NestJS payload
4. RAG Documents          ← facts retrieved this iteration
5. ToolMessage results    ← observations from prior loop iterations (most recent = last)
6. User Message           ← the current question
```

#### Key rules

- Long-term memory is **optional enrichment** — the loop must work correctly even when Milvus is unavailable.
- `user_memory` and `knowledge_base` are **separate Milvus collections**.
- The LLM never writes to memory directly — only the post-response step does.

---

### Component 6 — RAG Retriever

> For the full guide (chunking, ingestion pipeline, score tuning, hybrid search, re-ranking), see [agent-rag.md](agent-rag.md).

Retrieves external knowledge from the vector store when the question requires it.

**Pipeline:**

```text
User question
      │
      ▼
Embedding model (OpenAI text-embedding-3-small)
      │
      ▼
Milvus similarity search (knowledge_base collection)
      │
      ▼
Top-K document chunks retrieved
      │
      ▼
Filter by MIN_SCORE (discard low-relevance results)
      │
      ▼
Inject into ToolMessage as observation
```

**Implementation:**

```python
# app/rag/retriever.py
class RAGRetriever:
    TOP_K = 5
    MIN_SCORE = 0.72   # discard low-relevance results

    async def retrieve(self, query: str) -> list[Document]:
        embedding = await self.embedder.embed(query)

        results = await self.milvus_client.search(
            collection_name="knowledge_base",
            data=[embedding],
            limit=self.TOP_K,
            output_fields=["content", "source", "updated_at"],
        )

        return [
            Document(content=r["content"], source=r["source"], score=r["distance"])
            for r in results
            if r["distance"] >= self.MIN_SCORE
        ]
```

**Embedding model:**

```python
# app/rag/embeddings.py
class EmbeddingClient:
    # Claude does not provide embeddings — use a dedicated model
    MODEL = "text-embedding-3-small"    # OpenAI

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.MODEL,
            input=text
        )
        return response.data[0].embedding
```

---

### Component 7 — LLM Reasoning Engine

Claude is the core reasoning engine, accessed through **`ChatAnthropic`** from `langchain-anthropic`.

**Why `ChatAnthropic` instead of the raw Anthropic SDK:**
- `ChatAnthropic` speaks LangChain's `Runnable` interface — it accepts typed `HumanMessage`/`AIMessage`/`ToolMessage` objects, which is what `build_messages()` returns.
- In Phase 2 (LangGraph), `ChatAnthropic` is required so LangGraph can intercept LLM calls for streaming events. Using it from Phase 1 avoids a migration step.

**LLM client:**

```python
# app/llm/claude_client.py

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from app.config.settings import settings

# Module-level singleton — import `llm` directly in agent_loop.py
llm = ChatAnthropic(
    model=settings.claude_model,                  # "claude-sonnet-4-6"
    max_tokens=settings.claude_max_tokens,         # 4096
    anthropic_api_key=settings.anthropic_api_key,
    streaming=True,   # required for llm.astream() to yield tokens
)

# Convert history list of dicts → typed LangChain message objects.
# ChatAnthropic requires typed objects; the Anthropic raw SDK accepts plain dicts.
def build_messages(system_prompt: str, history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages
```

**Two call patterns:**

```python
# ── Pattern 1: planner call (wait for full response, parse tool_calls) ────────
planner  = llm.bind_tools(TOOLS)   # attach tool schemas to this call
response = await planner.ainvoke(messages)
# response.tool_calls: [{"name": "...", "args": {...}, "id": "..."}]
# response.content:    text reasoning (if Claude added any)

# ── Pattern 2: streaming final answer (yield tokens one-by-one) ───────────────
async for chunk in llm.astream(messages):
    yield chunk.content   # each chunk is a few characters; yield immediately as SSE
```

---

## 6. Token Budget Management

Agents that iterate multiple steps can silently overflow the Claude context window. Track tokens per request and trim history before each LLM call.

```python
# app/agent/token_budget.py

from dataclasses import dataclass

@dataclass
class TokenBudget:
    MODEL_CONTEXT_LIMIT = 200_000    # claude-sonnet-4-6 maximum context
    RESPONSE_RESERVE    = 4_096      # reserved for Claude's output
    SAFETY_MARGIN       = 2_000      # buffer for estimation errors

    USABLE_LIMIT = MODEL_CONTEXT_LIMIT - RESPONSE_RESERVE - SAFETY_MARGIN  # = 193,904

    used: int = 0

    def consume(self, tokens: int) -> None:
        self.used += tokens

    @property
    def remaining(self) -> int:
        return self.USABLE_LIMIT - self.used

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0

    def trim_history(self, history: list[dict]) -> list[dict]:
        """Drop oldest messages until the history fits the remaining budget."""
        # Drop in pairs (user + assistant) so conversation always starts with a user turn.
        while history and self._estimate(history) > self.remaining:
            history = history[2:]
        return history

    @staticmethod
    def _estimate(messages: list[dict]) -> int:
        # 1 token ≈ 4 characters (rough estimate for English)
        return sum(len(m["content"]) // 4 for m in messages)
```

**Budget is checked at three points:**

1. After parsing the request — estimate total incoming tokens
2. Before each `planner.ainvoke()` — trim history if needed
3. Before adding RAG documents — skip retrieval if no budget remains

---

## 7. The Agent Loop — `agent_loop.py`

This is the complete Phase 1 implementation. One file, one function, explicit control flow.

```python
# app/agent/agent_loop.py
#
# run() is an async generator: it yields SSE tokens one by one.
# The FastAPI router wraps it in StreamingResponse.
# The caller receives tokens as they are produced — no buffering.

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from app.llm.claude_client import llm, build_messages
from app.tools.registry import TOOLS, tool_map
from app.memory.vector_memory import VectorMemory
from app.rag.retriever import RAGRetriever
from app.schemas.request import AgentRequest
from app.agent.token_budget import TokenBudget

MAX_ITERATIONS = 10

async def run(request: AgentRequest):
    """
    Main agent loop. An async generator — yields tokens for SSE streaming.
    Called directly by the FastAPI router.
    """

    # ── Step 1: recall long-term memory ──────────────────────────────────────
    # Must happen BEFORE building messages so recalled context goes into the prompt.
    memory   = VectorMemory()
    recalled = await memory.recall(request.user_id, request.message, top_k=5)
    # recalled = list of MemoryChunk objects, e.g.:
    #   ["User prefers bullet points", "User works in finance"]
    # If Milvus is down, recall() returns [] — the loop still works.

    # ── Step 2: build initial message list ───────────────────────────────────
    system_prompt = build_system_prompt(recalled)
    # build_system_prompt() assembles:
    #   - agent identity ("You are a helpful assistant...")
    #   - long-term memory chunks (from recalled)
    #   - tool descriptions and usage rules
    #   - output format instructions

    messages = build_messages(system_prompt, [m.dict() for m in request.history])
    # build_messages() returns: [SystemMessage, HumanMessage, AIMessage, ...]
    # This covers all prior conversation turns.

    messages.append(HumanMessage(content=request.message))
    # Append the current user message as the final HumanMessage.

    # ── Step 3: set up planner and budget ────────────────────────────────────
    planner     = llm.bind_tools(TOOLS)
    # bind_tools() attaches JSON schemas for all tools to every ainvoke() call.
    # Claude uses these schemas to return structured tool_calls instead of plain text.

    budget      = TokenBudget()
    iterations  = []   # track tool calls and results for memory write later
    tokens_used = 0

    # ── Step 4: ReAct loop ───────────────────────────────────────────────────
    for i in range(MAX_ITERATIONS):

        # Trim history if token budget is running low
        messages = budget.trim_history_in_messages(messages)

        response     = await planner.ainvoke(messages)
        # ainvoke() waits for the FULL response — needed to read tool_calls.
        # response is an AIMessage with:
        #   response.content    = any text reasoning Claude added
        #   response.tool_calls = list of tool selections (empty if Claude is done)

        tokens_used += response.usage_metadata.get("total_tokens", 0)
        budget.consume(tokens_used)

        if budget.is_exhausted or i >= MAX_ITERATIONS - 1:
            # Budget or iteration cap reached — yield a graceful fallback
            yield "[Agent reached context limit. Partial answer follows.] "
            break

        if not response.tool_calls:
            # Claude returned no tool call → ready to give final answer.
            # Stream it token-by-token.
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
            break

        # ── Tool execution ────────────────────────────────────────────────────
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "rag_retrieve":
            # RAG is handled via the retriever, not the tool_map
            retriever = RAGRetriever()
            docs      = await retriever.retrieve(tool_args.get("query", request.message))
            result    = retriever.format_for_prompt(docs)
        else:
            tool = tool_map.get(tool_name)
            if tool is None:
                result = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result = await tool.ainvoke(tool_args)
                except Exception as e:
                    result = f"Error: {tool_name} failed — {str(e)}"
                    # Error becomes an observation — LLM sees it and adapts.

        iterations.append({
            "tool":   tool_name,
            "args":   tool_args,
            "result": str(result),
        })

        # Feed the result back so the planner sees it next iteration.
        messages.append(response)
        # ↑ Append the assistant turn (carries the tool_call reference).
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        # ↑ ToolMessage links this result to the tool_call via tool_call_id.
        #   The Anthropic API requires this pairing — without it, validation fails.

    # ── Step 5: write to long-term memory ────────────────────────────────────
    # Runs AFTER streaming, never during — adding latency to the token stream is bad UX.
    if iterations:
        summary = summarize_iterations(iterations)
        # summarize_iterations() condenses to 1-2 sentences, e.g.:
        #   "User asked about RAG benefits. Retrieved 4 documents. Provided 5-point answer."
        await memory.store_if_new(request.user_id, summary, tags=["session"])
```

---

## 8. Full Execution Flow

**User asks:** `"What are the benefits of RAG?"`

Steps execute as sequential function calls inside `agent_loop.run()`. State is held in local variables.

```text
① NestJS sends POST /v1/agent/chat
   { user_id, message, history, user_context, session_id }
   FastAPI parses body into AgentRequest (Pydantic)

② agent_loop.run(request) is called
   └── budget = TokenBudget()
   └── Estimates incoming token count → tokens_used = 1,240

③ guardrails.check(message)
   └── Content policy: PASS
   └── PII check: no PII detected → message unchanged
   └── Injection check: PASS

④ memory.recall(user_id, message, top_k=5)
   └── Returns 2 chunks: "User prefers bullet points", "User works in finance"
   └── Chunks injected into system_prompt string

⑤ build_messages(system_prompt, history) → messages list assembled
   messages.append(HumanMessage(content=request.message))
   planner = llm.bind_tools(TOOLS)

⑥ Loop iteration 1
   └── response = await planner.ainvoke(messages)
   └── response.tool_calls[0] → {"name": "rag_retrieve", "args": {"query": "benefits of RAG"}}
   └── tool_name == "rag_retrieve" → retriever.retrieve("benefits of RAG")
   └── result = 4 document chunks (scores 0.89–0.94, all above MIN_SCORE=0.72)
   └── messages.append(response)                           ← assistant turn (with tool_call)
   └── messages.append(ToolMessage(content=result, ...))   ← observation appended
   └── iterations.append({"tool": "rag_retrieve", ...})
   └── tokens_used += 820

⑦ Loop iteration 2
   └── response = await planner.ainvoke(messages)
   └── response.tool_calls == []  → no tool selected → break loop

⑧ Final answer streaming
   └── async for chunk in llm.astream(messages):
          yield chunk.content    ← each token sent as "data: <token>\n\n" SSE frame
   └── Browser renders tokens in real time

⑨ memory.store_if_new(user_id, summarize(iterations), tags=["session"])
   └── Written to Milvus user_memory collection

⑩ FastAPI yields "data: [DONE]\n\n" → NestJS receives stream end signal
   NestJS saves full response to PostgreSQL
```

---

## 9. Error Handling

| Failure Point             | Behaviour                                                            | User Impact                                               |
| ------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| Guardrails block          | `guardrails.check()` returns `passed=False` → FastAPI returns 422   | Immediate error message                                   |
| Tool error                | `try/except` in loop → error string becomes ToolMessage observation  | LLM sees error and adapts; may try a different tool       |
| RAG retrieval fails       | `retriever.retrieve()` returns `[]` → observation = "No docs found" | LLM answers from training data only                       |
| Token budget exhausted    | `budget.is_exhausted` check → yield fallback string, break loop     | User receives graceful fallback message                   |
| Max iterations reached    | `i >= MAX_ITERATIONS - 1` → yield fallback string, break loop       | User receives graceful fallback message                   |
| Claude API error          | `planner.ainvoke()` raises → exception propagates → NestJS 503      | Standard error response                                   |
| Pydantic validation error | FastAPI raises `RequestValidationError` before `run()` → 422        | Immediate schema error                                    |
| Milvus unreachable recall | `memory.recall()` catches error → returns `[]`                      | Agent continues without personalization                   |
| Milvus unreachable write  | `memory.store_if_new()` catches error → logs warning, continues     | Session completed; memory not persisted for next session  |

---

## 10. Project Folder Structure

Build all files in this order. The agent is fully functional when all of these exist.

```
app-ai/
├── app/
│   ├── main.py                          FastAPI app entry point, lifespan hooks
│   │
│   ├── routers/
│   │   ├── agent.py                     POST /v1/agent/chat → StreamingResponse SSE
│   │   └── rag.py                       POST /v1/rag/ingest, GET /v1/rag/ingest/{job_id}
│   │
│   ├── schemas/
│   │   ├── request.py                   AgentRequest, HistoryMessage, UserContext
│   │   └── response.py                  AgentResponse, Document
│   │
│   ├── guardrails/
│   │   ├── checker.py                   GuardrailChecker — orchestrates all checks
│   │   ├── content_policy.py            Policy violation detection
│   │   ├── pii_filter.py                PII redaction (regex + NER)
│   │   └── injection_detector.py        Prompt injection pattern matching
│   │
│   ├── agent/
│   │   ├── agent_loop.py                run() — async generator, manual for loop  ← KEY FILE
│   │   └── token_budget.py              TokenBudget class
│   │
│   ├── tools/
│   │   ├── registry.py                  TOOLS list + tool_map dict
│   │   ├── web_search.py                @tool: Tavily / SerpAPI integration
│   │   ├── code_exec.py                 @tool: sandboxed subprocess execution
│   │   ├── calculator.py                @tool: safe math evaluation
│   │   └── database.py                  @tool: read-only SQL query
│   │
│   ├── memory/
│   │   └── vector_memory.py             VectorMemory — long-term user preferences (Milvus)
│   │
│   ├── rag/
│   │   ├── retriever.py                 RAGRetriever — Milvus similarity search
│   │   ├── embeddings.py                EmbeddingClient (OpenAI text-embedding-3-small)
│   │   ├── chunker.py                   RecursiveCharacterTextSplitter wrapper
│   │   └── reranker.py                  Optional Cohere cross-encoder re-ranker
│   │
│   ├── llm/
│   │   └── claude_client.py             ChatAnthropic singleton + build_messages()
│   │
│   └── config/
│       └── settings.py                  Pydantic BaseSettings — all env vars
│
├── tests/
│   ├── unit/
│   │   ├── test_guardrails.py
│   │   ├── test_agent_loop.py           test run() with a mock LLM response
│   │   ├── test_token_budget.py
│   │   └── test_retriever.py
│   └── integration/
│       ├── test_agent_loop.py           run the full loop against a mock LLM
│       └── test_rag_pipeline.py
│
├── pyproject.toml
├── Dockerfile
└── .env.example
```

---

## 11. FastAPI Endpoint — Streaming SSE

The router calls `run()` directly. `run()` is an async generator that yields tokens. The router wraps it in `StreamingResponse`.

```python
# app/routers/agent.py

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.request import AgentRequest
from app.guardrails.checker import GuardrailChecker
from app.agent.agent_loop import run

router   = APIRouter(prefix="/v1")
guardrails = GuardrailChecker()

@router.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    # Guardrails run before the loop — rejected requests never enter run().
    guard_result = guardrails.check(request.message)
    if not guard_result.passed:
        raise HTTPException(status_code=422, detail=guard_result.reason)

    # Replace the raw message with the sanitized version (PII redacted).
    request.message = guard_result.sanitized_message

    async def token_stream():
        # run() is an async generator — yields one token string per iteration.
        # Each yield becomes one SSE frame sent to the browser immediately.
        async for token in run(request):
            if token:
                yield f"data: {token}\n\n"   # SSE format: "data: <content>\n\n"
        yield "data: [DONE]\n\n"             # signals the browser the stream is finished

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",    # every token must reach the client immediately
            "X-Accel-Buffering": "no",      # tells Nginx: proxy_buffering off (required for SSE)
        }
    )
```

---

## 12. Configuration

```python
# app/config/settings.py

from pydantic_settings import BaseSettings   # pip install pydantic-settings

class Settings(BaseSettings):
    # ── Claude ────────────────────────────────────────────────────────────────
    anthropic_api_key: str            # REQUIRED — app won't start without it
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 4096

    # ── Embeddings ────────────────────────────────────────────────────────────
    openai_api_key: str               # for text-embedding-3-small
    embedding_model: str = "text-embedding-3-small"

    # ── Milvus ────────────────────────────────────────────────────────────────
    milvus_host: str = "milvus"       # matches Docker Compose service name
    milvus_port: int = 19530
    milvus_collection_knowledge: str = "knowledge_base"  # shared RAG documents
    milvus_collection_memory:    str = "user_memory"     # per-user long-term memory

    # ── Agent loop ────────────────────────────────────────────────────────────
    agent_max_iterations: int = 10
    rag_top_k: int = 5
    rag_min_score: float = 0.72

    # ── Optional ──────────────────────────────────────────────────────────────
    tavily_api_key: str | None = None   # agent works without web search

    class Config:
        env_file = ".env"

settings = Settings()
```

```ini
# .env.example
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
MILVUS_HOST=milvus
MILVUS_PORT=19530
TAVILY_API_KEY=tvly-...
```

---

## 13. Observability

Track every iteration of the agent loop with Langfuse.

```python
# app/agent/agent_loop.py  (with observability)

from langfuse import Langfuse

langfuse = Langfuse()   # reads LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY from env

async def run(request: AgentRequest):
    trace = langfuse.trace(
        name="agent_loop",
        user_id=request.user_id,
        session_id=request.session_id,
        input={"message": request.message},
    )

    for i in range(MAX_ITERATIONS):
        span = trace.span(name=f"iteration_{i}")

        # ... planner.ainvoke(), tool execution, ToolMessage append ...

        span.end(output={
            "tool":        tool_name if response.tool_calls else "final_answer",
            "tokens_used": tokens_used,
        })

    trace.update(output={"streamed": True})
```

**What to track per iteration:**

- Tool name and input/output
- RAG query, number of documents returned, scores
- Token count at each step
- Wall clock time per iteration

---

## 14. Production Enhancements

### Streaming passthrough

Every layer must be configured to not buffer SSE responses:

| Layer   | Required configuration                                           |
| ------- | ---------------------------------------------------------------- |
| FastAPI | `StreamingResponse(generator, media_type="text/event-stream")`   |
| NestJS  | `upstream.data.pipe(res)` — never `await` the full response      |
| Nginx   | `proxy_buffering off; proxy_cache off; proxy_read_timeout 300s;` |

### Tool sandboxing

Code execution tools must run in an isolated subprocess:

```python
# app/tools/code_exec.py
#
# SECURITY CRITICAL: User-supplied code must NEVER run in the same process as the agent.

import subprocess, resource

result = subprocess.run(
    ["python3", "-c", user_code],
    capture_output=True,
    timeout=10,
    preexec_fn=lambda: resource.setrlimit(
        resource.RLIMIT_AS,
        (256 * 1024 * 1024, 256 * 1024 * 1024)   # 256 MB memory limit
    )
)
```

### Async background tasks (long-running jobs)

```python
# app/routers/rag.py
from fastapi import BackgroundTasks

@router.post("/v1/rag/ingest")
async def ingest_document(file: UploadFile, background_tasks: BackgroundTasks):
    job_id = generate_job_id()
    background_tasks.add_task(embedding_pipeline.run, file, job_id)
    return {"job_id": job_id, "status": "queued"}
```

---

## 15. Key Architectural Principles

1. **Stateless by design.** `app-ai` receives full context on every call. No in-memory state survives between requests. All persistent state lives in NestJS/PostgreSQL (conversation) and Milvus (vectors).

2. **Guardrails run before the LLM.** They cannot be bypassed by any tool call or agent action.

3. **Tool failures return observations, not exceptions.** A failed tool produces a string that the LLM receives as an observation. The loop continues. Only infrastructure failures (Claude API down, Milvus unreachable) raise exceptions.

4. **Token budget is enforced, not estimated.** History is actively trimmed before every LLM call. The agent never assumes it has budget remaining.

5. **Max iterations is a hard limit.** When reached, the agent yields a graceful fallback. Runaway loops are not retried.

6. **`app-ai` never validates auth.** NestJS validates JWT and passes only `user_id`. No auth logic belongs in Python.

7. **Long-term memory write happens after streaming.** Never during the loop — adding Milvus write latency to the token stream is bad UX.

---

## 16. Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python             = "^3.11"
fastapi            = "^0.111"
uvicorn            = "^0.29"
pydantic           = "^2.7"
pydantic-settings  = "^2.2"
langchain-anthropic = "^0.1"    # ChatAnthropic
langchain-core     = "^0.3"     # @tool, typed messages
pymilvus           = "^2.4"     # Milvus client
openai             = "^1.30"    # text-embedding-3-small
langchain-text-splitters = "^0.2"  # RecursiveCharacterTextSplitter (for RAG ingestion)
langfuse           = "^2.0"     # observability (optional)
tavily-python      = "^0.3"     # web search (optional)
```

```bash
pip install fastapi uvicorn pydantic pydantic-settings \
            langchain-anthropic langchain-core \
            pymilvus openai langchain-text-splitters
```

---

## 17. Next Step — Upgrade to LangGraph

Phase 1 (`agent_loop.py`) is a complete, working agent. Once it is stable, you can upgrade the loop internals to a LangGraph `StateGraph` without changing any external API or tool code.

**What changes in Phase 2:**

| Phase 1 (this document)              | Phase 2 (LangGraph)                                      |
| ------------------------------------ | -------------------------------------------------------- |
| `for i in range(MAX_ITERATIONS)`     | `route_after_planner()` + `add_conditional_edges`        |
| Local `messages` list                | `AgentState["history"]` with `add_messages` reducer      |
| `messages.append(ToolMessage(...))`  | Node returns partial state; LangGraph merges             |
| Local `iterations` list              | `AgentState["iterations"]` updated by each node function |
| `llm.astream()` called in `run()`   | `llm.astream()` inside `llm_respond_node`                |
| No crash recovery                    | `MemorySaver` checkpoints state after every node         |

**Files that stay the same in Phase 2 (no changes needed):**

```text
✅ app/guardrails/       — all guardrail files unchanged
✅ app/tools/            — all @tool files unchanged
✅ app/rag/              — all RAG files unchanged
✅ app/memory/vector_memory.py — unchanged
✅ app/config/settings.py     — unchanged
✅ app/schemas/               — unchanged
```

**Files replaced in Phase 2:**

```text
🔄 agent/agent_loop.py  → agent/state.py + agent/nodes.py + agent/graph.py
🔄 routers/agent.py     → small update: astream_events instead of run()
```

See [app-ai-with-langgraph.md](app-ai-with-langgraph.md) for the full Phase 2 guide.

---

## 18. Cross-references

| Topic                                            | Document                                                             |
| ------------------------------------------------ | -------------------------------------------------------------------- |
| Full system architecture (all tiers)             | [app-architecture.md](app-architecture.md)                           |
| Memory deep-dive (scopes, rules, GDPR)           | [agent-memory.md](agent-memory.md)                                   |
| RAG deep-dive (chunking, ingestion, re-ranking)  | [agent-rag.md](agent-rag.md)                                         |
| Phase 2 — LangGraph upgrade                      | [app-ai-with-langgraph.md](app-ai-with-langgraph.md)                 |
| NestJS → FastAPI communication                   | [app-architecture.md § Communication Protocols](app-architecture.md) |
| Streaming configuration (Nginx, NestJS, FastAPI) | [app-architecture.md § Streaming rule](app-architecture.md)          |
