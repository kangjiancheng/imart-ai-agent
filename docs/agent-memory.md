# Agent Memory System

> **Scope:** Deep-dive guide for the memory layer inside `app-ai`.
> For the full agent architecture, see [app-ai-architecture.md](app-ai-architecture.md).

---

## 1. What Is Memory in an AI Agent?

An LLM on its own is **stateless** — every call starts from a blank slate. Without memory, an agent cannot:

- Remember what it decided three steps ago in the same task
- Recall that this user prefers concise answers
- Know that it already tried one tool and it failed
- Understand the context of a follow-up question like *"Can you expand on that?"*

Memory is the mechanism that gives an agent **continuity**. It is not a single thing — it is a set of layered stores, each with a different scope, lifetime, and purpose.

```text
Without memory:            With memory:

User: "Find Tesla revenue" User: "Find Tesla revenue"
Agent: searches → answers  Agent: searches → answers
                                 ↓ stores observation
User: "Compare with Ford"  User: "Compare with Ford"
Agent: ??? (no context)    Agent: recalls Tesla result → compares
```

---

## 2. Why Memory Matters

### The LLM is not the memory

The LLM has a context window — a fixed token limit per call. Everything the LLM "knows" during a single call must be injected into that window at call time. The LLM does not retain anything between calls. Memory is the system that **decides what to inject and when**.

### Three problems memory solves

| Problem | Without memory | With memory |
|---|---|---|
| Multi-step reasoning | Each LLM call is isolated; agent re-derives state | Prior thoughts and observations are injected; reasoning is coherent |
| Repeated tool calls | Agent may call the same tool twice for the same data | Tool result stored; second call skipped |
| Personalization | Every session starts cold; user must repeat preferences | User preferences recalled from vector store |

---

## 3. The Two Memory Scopes

`app-ai` uses two distinct memory scopes. They serve different purposes and have different lifetimes.

```text
┌─────────────────────────────────────────────────────────┐
│                    ONE REQUEST                          │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │     Short-term Memory  →  AgentState (LangGraph)  │  │
│  │                                                   │  │
│  │  Node 1: planner  → tool_executor (observation)  │  │
│  │  Node 2: planner  → rag_retrieve  (observation)  │  │
│  │  Node 3: planner  → llm_respond   (final answer) │  │
│  │                                                   │  │
│  │  Lives in LangGraph state. Checkpointed by        │  │
│  │  MemorySaver. Scoped to one thread_id / session.  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 ACROSS SESSIONS                         │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │             Long-term Memory (Milvus)             │  │
│  │                                                   │  │
│  │  User preferences, past task summaries,           │  │
│  │  domain knowledge specific to this user.          │  │
│  │                                                   │  │
│  │  Lives in vector DB. Persists indefinitely.       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Short-term Memory — `AgentState` (LangGraph)

### What it is

Short-term memory is the **state of the LangGraph `StateGraph` for one request**. LangGraph manages it automatically as data flows between nodes. Each node receives the current state as input and returns a partial update — LangGraph merges the update and passes the new state to the next node.

In the old manual loop, a `ShortMemory` dataclass was created at the start and passed around by hand. LangGraph replaces this entirely with a `TypedDict` called `AgentState`.

### What it stores

```python
# app/agent/state.py
#
# AgentState is the LangGraph state schema — a TypedDict that defines every field
# the agent tracks across nodes within one request.
# LangGraph automatically passes this state to each node and merges node outputs back in.
# No manual "record()" calls needed — LangGraph handles state propagation.

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages   # reducer: appends new messages to the list

class AgentState(TypedDict):
    # --- identity ---
    user_id:     str        # which user this request belongs to (used for memory recall/write)
    session_id:  str        # maps to LangGraph thread_id for MemorySaver checkpointing

    # --- message history ---
    # Annotated[list, add_messages] tells LangGraph to APPEND new messages, not replace the list.
    # This is the standard LangGraph pattern for conversation history.
    history: Annotated[list, add_messages]

    # --- per-request tracking (replaces ShortMemory fields) ---
    iterations:     list[dict]   # one dict per planner→node round trip (replaces Iteration dataclass)
    tool_results:   dict         # keyed by tool_name — deduplicates calls (same purpose as before)
    rag_documents:  list[str]    # documents retrieved this request (same purpose as before)
    tokens_used:    int          # running token total (replaces TokenBudget)

    # --- routing ---
    current_action: dict         # set by planner_node: {"name": "tool_call", "args": {...}}
                                 # read by route_after_planner() to decide next node

    # --- context injected before loop ---
    long_term_memories: list[str]   # recalled from Milvus before planner starts
    error:              str | None  # set by guardrails if request is blocked
```

### How LangGraph updates state between nodes

Each node returns only the fields it changed. LangGraph merges the return value into the current state automatically.

```python
# app/agent/nodes.py  (excerpt)

async def tool_executor_node(state: AgentState) -> dict:
    # Read the action the planner decided
    action     = state["current_action"]
    tool_name  = action["name"]
    tool_args  = action.get("args", {})

    # Execute the tool
    tool       = tool_map.get(tool_name)
    observation = await tool.ainvoke(tool_args)   # @tool-decorated function, async call

    # Return ONLY the fields this node changes — LangGraph merges the rest
    return {
        "tool_results": {**state["tool_results"], tool_name: observation},
        # Append a new iteration record to the list
        "iterations": state["iterations"] + [{
            "node":        "tool_executor",
            "tool":        tool_name,
            "args":        tool_args,
            "observation": observation,
        }],
        "tokens_used": state["tokens_used"] + count_tokens(str(observation)),
    }
    # LangGraph passes the merged state to the next node (back to planner)
    # No ShortMemory.record() call needed — state propagation is automatic
```

### Why it matters in the loop

Without state propagation, `planner_node` on the second call would have no idea what the tool returned on the first call. LangGraph guarantees that every node sees the full accumulated state.

```text
Node: planner (call 1)
  current_action = {"name": "web_search", "args": {"query": "Tesla revenue 2024"}}
  → LangGraph routes to tool_executor

Node: tool_executor
  observation = "Tesla Q4 2024 revenue was $25.7B."
  → state["iterations"] updated; state["tool_results"] updated
  → LangGraph routes back to planner

Node: planner (call 2)
  [full AgentState injected — planner sees the tool result in history]
  current_action = {"name": "final_answer", "args": {}}
  → LangGraph routes to llm_respond
```

### Lifetime and checkpointing

`AgentState` is created fresh at the start of each request (passed as `initial_state` to `agent_graph.astream_events()`). `MemorySaver` checkpoints the state after every node completes. If the server restarts mid-request, the graph can resume from the last checkpoint using the same `thread_id`.

State is scoped to one `thread_id` (= `session_id`). It does not persist beyond the session unless explicitly written to long-term memory (Milvus).

---

## 5. Long-term Memory

### What it is

Long-term memory is a **user-scoped vector store** in Milvus. It holds information that should survive across sessions — things the agent should remember the next time this user talks to it.

### What it stores

```text
User preferences
└── "Prefers bullet-point summaries over paragraphs"
└── "Always wants answers in Korean"
└── "Works in finance — use industry terminology"

Past task summaries
└── "On 2024-12-01, researched Tesla vs Ford revenue comparison"
└── "User's project is building an AI-powered equity research tool"

Domain-specific context
└── Custom knowledge uploaded by the user
└── Company-specific documents indexed for this user
```

### How recall works

```python
# app/memory/vector_memory.py
#
# VectorMemory is the long-term store: it survives across sessions by persisting to Milvus.
# It uses the SAME embedding + vector search approach as RAG, but in the `user_memory`
# collection (per-user, GDPR-scoped) instead of `knowledge_base` (shared across all users).

from app.rag.embeddings import EmbeddingClient   # reuse the same embedding client as RAG
from app.config.settings import settings
from pymilvus import MilvusClient

# Data container for one recalled memory chunk
@dataclass
class MemoryChunk:
    content:    str
    score:      float       # similarity score from the vector search
    stored_at:  str         # ISO timestamp: when this memory was written
    tags:       list[str]   # e.g. ["preference", "format"] or ["context", "domain"]

class VectorMemory:
    COLLECTION = "user_memory"   # separate collection from knowledge_base — different GDPR scope
    MIN_SCORE  = 0.78            # higher threshold than RAG (0.72) — memory recall must be confident
    #                              A wrong memory is worse than no memory at all

    def __init__(self):
        self.milvus   = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        self.embedder = EmbeddingClient()

    async def store(self, user_id: str, content: str, tags: list[str] = []) -> None:
        # Step 1: embed the memory content into a vector
        embedding = await self.embedder.embed(content)
        # Step 2: insert one record into Milvus
        # `data` is a list of dicts — each dict is one row to insert
        self.milvus.insert(
            collection_name=self.COLLECTION,
            data=[{
                "vector":    embedding,
                "user_id":   user_id,       # partitions this memory to its owner
                "content":   content,
                "tags":      tags,
                "stored_at": datetime.utcnow().isoformat(),   # e.g. "2025-01-15T10:23:45"
            }]
        )

    async def recall(self, user_id: str, query: str, top_k: int = 5) -> list[MemoryChunk]:
        # Embed the current user message to find semantically similar past memories
        query_embedding = await self.embedder.embed(query)
        results = self.milvus.search(
            collection_name=self.COLLECTION,
            data=[query_embedding],
            # IMPORTANT: filter by user_id so users never see each other's memories
            filter=f"user_id == '{user_id}'",
            limit=top_k,
            output_fields=["content", "stored_at", "tags"],
        )
        # Build MemoryChunk objects, discarding low-confidence matches
        return [
            MemoryChunk(
                content=r["entity"]["content"],
                score=r["distance"],
                stored_at=r["entity"]["stored_at"],
                tags=r["entity"]["tags"],
            )
            for r in results[0]
            if r["distance"] >= self.MIN_SCORE   # filter: only confident matches
        ]

    async def forget(self, user_id: str, memory_id: str) -> None:
        """GDPR / user-initiated deletion — removes a specific memory record."""
        # Milvus `delete` accepts a filter expression (like SQL WHERE)
        self.milvus.delete(
            collection_name=self.COLLECTION,
            filter=f"user_id == '{user_id}' and id == '{memory_id}'"
            # Two conditions joined with `and` — only deletes the matching user's record
        )
```

### When to write vs when to read

```text
READ  (recall)  — at the start of each request, before the Planner runs
                  inject recalled chunks into the LLM system prompt
                  gives the agent user context before it begins reasoning

WRITE (store)   — at the end of a successful request
                  summarize what was useful: preferences expressed,
                  decisions made, topics explored
                  never write during the agent loop — only after final answer
```

---

## 6. Memory vs RAG — The Critical Distinction

These two components both use Milvus and embeddings, but they serve fundamentally different purposes.

| Dimension | Long-term Memory | RAG (Knowledge Base) |
|---|---|---|
| **Owner** | One user | Shared across all users |
| **Content** | User preferences, past tasks | Company documents, product manuals, public knowledge |
| **Written by** | The agent (after each session) | Ingestion pipeline (one-time or scheduled) |
| **Milvus collection** | `user_memory` | `knowledge_base` |
| **Recall threshold** | 0.78 (high — must be confident) | 0.72 (lower — broader retrieval acceptable) |
| **Purpose** | Personalization and continuity | Factual grounding for answers |
| **GDPR scope** | Per-user deletion required | Document-level deletion |

```text
RAG answers:     "What does our refund policy say?"
                  → retrieves from shared knowledge_base

Memory answers:  "What did this user tell us about their payment preferences?"
                  → retrieves from user_memory
```

They are often used **together** in the same request:

```text
LLM prompt =  system prompt
           +  [long-term memory]   ← what we know about THIS user
           +  conversation history
           +  [RAG documents]      ← facts retrieved for THIS question
           +  short-term memory    ← what happened in THIS loop so far
           +  user message
```

---

## 7. How Memory Fits in the Workflow

```text
① NestJS sends request
        │
        ▼
② input_handler node — normalizes payload → sets initial AgentState

③ guardrails node — safety checks
   └── sets state["error"] if blocked → graph routes to END
        │
        ▼
④ memory_recall node  (Long-term Memory RECALL)
   └── recall(user_id, message, top_k=5)
   └── state["long_term_memories"] populated
   └── agent now "knows" this user before reasoning starts
        │
        ▼
⑤ LangGraph StateGraph loop begins
   │
   ├── planner node (call 1)
   │     ChatAnthropic.bind_tools(TOOLS).ainvoke(messages)
   │     state["current_action"] = {"name": "rag_retrieve", "args": {...}}
   │     route_after_planner() → "rag_retrieve"
   │
   ├── rag_retrieve node
   │     retrieves documents → state["rag_documents"] updated
   │     state["iterations"] appended with observation
   │     → graph edges back to planner
   │
   ├── planner node (call 2)
   │     [full AgentState injected — planner sees rag_retrieve result]
   │     state["current_action"] = {"name": "final_answer", "args": {}}
   │     route_after_planner() → "llm_respond"
   │
   └── llm_respond node — streams final answer via astream_events()
        │
        ▼
⑥ memory_write node  (Long-term Memory WRITE, after stream ends)
   └── summarize useful context from this session
   └── store: preferences expressed, topics covered, decisions made
   └── next session starts with this context pre-loaded
        │
        ▼
       END
```

### Memory injection point in the LLM prompt

LangGraph calls `build_messages()` inside `planner_node` to assemble the full message list before each `ChatAnthropic.ainvoke()` call.

```text
Position 1  →  SystemMessage         (agent identity, tool list, rules)
Position 2  →  Long-term Memory      ← recalled in memory_recall node, injected here
Position 3  →  Conversation History  (HumanMessage / AIMessage from AgentState["history"])
Position 4  →  RAG Documents         (retrieved in rag_retrieve node, from AgentState["rag_documents"])
Position 5  →  Prior Iterations      (tool results / observations in AgentState["iterations"])
Position 6  →  HumanMessage          (current user message)
```

Position matters. Long-term memory is placed early so it shapes how the LLM interprets everything that follows. Prior iterations are placed late so the most recent observations are closest to the LLM's "attention".

---

## 8. What Gets Written to Long-term Memory

Not every session should write to memory. Write only when there is signal worth keeping.

**Write these:**

```text
✅ Explicit user preference
   "Please always give me bullet points"
   → store: { content: "User prefers bullet-point format", tags: ["preference", "format"] }

✅ Domain context revealed
   "I'm building an equity research tool"
   → store: { content: "User is building an equity research AI tool", tags: ["context", "domain"] }

✅ Task summary (long or complex tasks)
   After a multi-step research task:
   → store: { content: "Researched Tesla vs Ford revenue 2024. Tesla: $25.7B, Ford: $18.5B", tags: ["research", "finance"] }

✅ User correction of agent behaviour
   "Don't use jargon — I'm not a finance expert"
   → store: { content: "User is not a finance expert. Avoid jargon.", tags: ["preference", "tone"] }
```

**Do not write these:**

```text
❌ Every message (creates noise, degrades recall quality)
❌ Transient tool results (already captured in AgentState["tool_results"])
❌ Failed or aborted sessions
❌ Sensitive PII (email, phone, payment info — guardrails should have caught this)
❌ Duplicate information already in the vector store
```

**Deduplication check before writing:**

```python
# Deduplication guard — called before every memory write.
# Without this, every session would add a new "User prefers bullet points" record
# and retrieval would return 50 copies of the same fact, wasting tokens.

async def store_if_new(self, user_id: str, content: str, tags: list[str]) -> None:
    # Recall the single most similar existing memory (top_k=1)
    existing = await self.recall(user_id, content, top_k=1)

    # If the best match is ≥ 0.95 similarity, it's essentially the same content — skip the write.
    # 0.95 is much higher than the 0.78 recall threshold: we only skip if it's nearly identical.
    if existing and existing[0].score >= 0.95:
        return    # near-duplicate detected — abort to avoid redundant storage

    await self.store(user_id, content, tags)   # new enough to be worth storing
```

---

## 9. Memory Failure Modes and Handling

| Failure | Behaviour | Impact |
|---|---|---|
| Milvus unreachable at recall | Skip long-term memory injection, continue without it | Agent works without personalization; no crash |
| Milvus unreachable at write | Log warning, discard write | Session completed normally; memory not persisted |
| Recall returns zero results | Empty list injected; LLM proceeds with history only | Normal behaviour for first-time users |
| Recall score below threshold | Chunk discarded; not injected | Prevents low-confidence hallucinations |
| Write creates duplicate | Deduplication check prevents it | No wasted storage |
| User requests deletion (GDPR) | `forget(user_id, memory_id)` removes from Milvus | Compliance met |

Long-term memory is **optional enrichment**, never a required dependency. The agent loop must complete correctly even when Milvus is unavailable.

---

## 10. Memory Design Rules

1. **Short-term memory is `AgentState` — managed by LangGraph, not application code.** LangGraph checkpoints the state (via `MemorySaver`) after every node. If the process crashes mid-request, the graph can resume from the last checkpoint using the same `thread_id`. State is scoped to one session and does not leak across users.

2. **Long-term memory is written after the response is sent**, never during the agent loop. Writing during the loop would add latency to the user-visible stream.

3. **Long-term memory recall happens once per request**, before the Planner runs. Recalling mid-loop is wasteful and adds latency.

4. **Memory and RAG use separate Milvus collections.** `user_memory` is per-user and GDPR-scoped. `knowledge_base` is shared and not subject to per-user deletion.

5. **Memory content must be summarized, not raw.** Raw LLM outputs are too long and too noisy to store directly. Summarize to one or two sentences before writing.

6. **The LLM does not write to memory directly.** Only the post-response handler writes to long-term memory. The LLM can suggest what to remember (as a structured output field), but the write decision is always made in application code.

---

## 11. Files

```
app-ai/
└── app/
    ├── agent/
    │   ├── state.py        AgentState TypedDict — short-term memory schema (replaces short_memory.py)
    │   ├── nodes.py        One async node function per graph step; each returns partial state update
    │   └── graph.py        StateGraph construction, routing, MemorySaver checkpointing
    └── memory/
        └── vector_memory.py    Long-term user-scoped store (Milvus read/write/forget)
```

> `short_memory.py` has been removed. `AgentState` in `agent/state.py` replaces it entirely.
> LangGraph propagates state between nodes automatically — no manual `.record()` calls.

Cross-references:
- [app-ai-architecture.md — Component 5](app-ai-architecture.md) — AgentState and short-term memory in the full agent
- [app-ai-architecture.md — Component 6](app-ai-architecture.md) — RAG Retriever (uses same Milvus, different collection)
- [app-ai-architecture.md — Section 6](app-ai-architecture.md) — LangGraph StateGraph construction and routing
- [app-ai-architecture.md — Section 7](app-ai-architecture.md) — LLM prompt assembly order (`build_messages()`)
