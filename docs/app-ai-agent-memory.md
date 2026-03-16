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

## 2. The Three Memory Layers

`app-ai` uses three distinct memory layers that work together on every request.

```text
┌─────────────────────────────────────────────────────────┐
│                    ONE REQUEST                          │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 1: Conversation History                    │  │
│  │                                                   │  │
│  │  HumanMessage / AIMessage / ToolMessage turns     │  │
│  │  sent by the client (NestJS) + built during loop  │  │
│  │  Injected as the main message list to Claude.     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 2: RAG (knowledge_base)                    │  │
│  │                                                   │  │
│  │  Company documents, manuals, policies.            │  │
│  │  Retrieved mid-loop when Claude calls rag_retrieve│  │
│  │  Injected as a ToolMessage observation.           │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 ACROSS SESSIONS                         │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 3: Vector Memory (user_memory)             │  │
│  │                                                   │  │
│  │  User preferences, past task summaries.           │  │
│  │  Per-user, persists across sessions in Milvus.    │  │
│  │  Recalled at request start, injected in system    │  │
│  │  prompt before Claude sees anything.              │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1 — Conversation History

### What it is

The raw list of typed message objects that Claude reads as the conversation transcript. It is assembled fresh on every request from:

1. Prior turns that NestJS sends in `request.history` (already serialized from the previous session)
2. New turns appended during the ReAct loop as tools are called and results come back

### What it stores

```python
# Built in src/llm/claude_client.py → build_messages()
# Converted from plain dicts to typed LangChain objects

[
    SystemMessage("You are a helpful AI assistant..."),  # always position 0
    HumanMessage("What is 15% of $340?"),               # from request.history
    AIMessage("Let me calculate that."),                 # from request.history
    HumanMessage("Now add tax."),                        # current message
]

# Then the ReAct loop appends during execution:
    AIMessage(tool_calls=[{name: "calculator", args: {...}}]),
    ToolMessage("51.0", tool_call_id="tc_001"),
```

### Lifetime

Created at the start of each request. Discarded when the request ends. The client (NestJS) is responsible for persisting it between sessions and sending it back in the next request.

### Workflow inside the ReAct loop

```text
POST /v1/agent/chat arrives
{message: "Now also add 8% tax.", history: [...], user_id: "u1"}
        │
        ▼
① build_messages(system_prompt, request.history)          ← claude_client.py
  converts request.history dicts → typed LangChain objects
  result: [SystemMessage, HumanMessage, AIMessage, ...]
  ↑ this is the "history" portion — what NestJS sent back

② messages.append(HumanMessage(request.message))         ← agent_loop.py:118
  adds the current user turn at the end
  ↑ messages list is now fully assembled

        │
        ▼  ReAct loop starts (up to 10 iterations)
        │
③ messages = budget.trim_history(messages)               ← every iteration
  drops oldest turns if approaching 193,904 token limit
  SystemMessage is NEVER dropped — only history turns

④ response = await planner.ainvoke(messages)             ← sends full list to Claude
  Claude reads ALL messages top-to-bottom before deciding

        ├── Claude has tool_calls → execute tool
        │
        │   messages.append(response)                    ← AIMessage with tool_call
        │   messages.append(ToolMessage(result, id))     ← tool result
        │   ↑ list grows by 2 every iteration
        │
        │   loop back to ③ — Claude now sees the tool result
        │
        └── Claude has no tool_calls → stream final answer
                planner.astream(messages)                ← same list, streaming mode
                break
        │
        ▼
⑤ Request ends — messages list is garbage collected
  Nothing is saved server-side.
  NestJS reads the streamed answer and stores it in its own DB
  to send back as request.history on the next request.
```

### Token management

The message list grows with every tool call. `TokenBudget` (`src/agent/token_budget.py`) tracks usage and trims the oldest turns when the 200,000-token context window runs low:

```text
Model context limit:  200,000 tokens
  - Response reserve:   4,096 tokens  (room for Claude's answer)
  - Safety margin:      2,000 tokens  (estimate error buffer)
  = Usable input limit: 193,904 tokens

When approaching the limit:
  trim_history() drops the oldest user+tool turn block,
  always keeping the SystemMessage intact.
```

---

## 4. Layer 2 — RAG (Retrieval-Augmented Generation)

### What it is

RAG lets Claude answer questions about documents that were not in its training data — product manuals, internal policies, API references, etc. Documents are chunked, embedded into vectors, and stored in Milvus. At query time, the semantically closest chunks are retrieved and shown to Claude as tool results.

### How it works

```text
WRITE path (one-time, before users chat):
  POST /v1/rag/ingest  →  chunk_text(doc)
                       →  embedder.embed(chunk)  per chunk
                       →  MilvusClient.insert() → "knowledge_base" collection

READ path (during agent loop):
  Claude decides to call rag_retrieve("query")
    →  embedder.embed(query)           convert query to vector
    →  MilvusClient.search()           find TOP_K nearest vectors
    →  filter: score >= MIN_SCORE 0.72 discard weak matches
    →  RAGRetriever.format_for_prompt() build readable string
    →  ToolMessage injected into messages list
    →  Claude reads it and answers
```

### Key parameters

| Parameter | Value | Purpose |
|---|---|---|
| `TOP_K` | 5 | Max results returned from Milvus per search |
| `MIN_SCORE` | 0.72 | Cosine similarity threshold — below this is discarded |
| Embedding model | BGE-M3 (local) | 1024-dim vectors, no API cost |
| Milvus collection | `knowledge_base` | Shared across all users |

### Source

`src/rag/retriever.py` — `RAGRetriever.retrieve()` and `format_for_prompt()`

---

## 5. Layer 3 — Long-term Vector Memory

### What it is

Long-term memory is a **user-scoped vector store** in Milvus. It holds information that should survive across sessions — things the agent should remember the next time this user talks to it. It uses the exact same embedding + vector search technique as RAG, but in a separate `user_memory` collection filtered strictly by `user_id`.

### What it stores

```text
User preferences
└── "User prefers bullet-point summaries over paragraphs"
└── "Always wants answers in Korean"
└── "Works in finance — use industry terminology"

Past task summaries (written automatically after each session)
└── "Agent used 2 tool call(s): rag_retrieve, calculator. Last result: 51.0"
└── "Agent used 1 tool call(s): web_search. Last result: Tesla Q4 2024..."
```

### When to read vs write

```text
READ  (recall)  — BEFORE the loop starts, before Claude sees anything
                  inject recalled chunks into the LLM system prompt
                  gives the agent user context before it begins reasoning

WRITE (store)   — AFTER streaming is complete (never during the loop)
                  summarizes what tools were used and what was found
                  only written when tool calls actually happened
                  writing during the loop would add latency to the stream
```

### How recall works

```python
# src/memory/vector_memory.py — VectorMemory.recall()

# 1. Embed the current user message
embedding = await self.embedder.embed(query)

# 2. Search Milvus — filtered strictly to this user
results = client.search(
    collection_name="user_memory",
    data=[embedding],
    filter=f'user_id == "{user_id}"',   # ← critical: scopes to one user
    limit=top_k,                         # default 5
    output_fields=["content"],
)

# 3. Filter by confidence threshold
chunks = [
    hit["entity"]["content"]
    for hit in results[0]
    if hit.get("distance", 0) >= 0.78   # higher than RAG's 0.72 — must be confident
]
```

### Why a higher threshold than RAG (0.78 vs 0.72)?

A wrong memory injected into the system prompt can significantly mislead Claude before it even starts reasoning. A wrong RAG document at least arrives as a tool result that Claude can reason about and discard. It is better to retrieve nothing than the wrong personal fact.

### How write works

```python
# src/memory/vector_memory.py — VectorMemory.store_if_new()
# Called from agent_loop.py AFTER streaming ends

# src/agent/agent_loop.py — _summarize_iterations()
# Condenses the tool call log into 1-2 sentences:
# "Agent used 2 tool call(s): rag_retrieve, calculator. Last tool result snippet: 51.0"

await memory.store_if_new(request.user_id, summary, tags=["session"])
```

### Key parameters

| Parameter | Value | Purpose |
|---|---|---|
| `MIN_SCORE` | 0.78 | Recall threshold — above this is injected |
| `top_k` | 5 | Max memories to recall per request |
| Milvus collection | `user_memory` | Per-user, GDPR-scoped |
| Filter | `user_id == "..."` | Ensures users never see each other's memories |

---

## 6. Short-term vs Long-term Memory

### Side-by-side comparison

| | Short-term (Conversation History) | Long-term (Vector Memory) |
|---|---|---|
| **What it stores** | Full message transcript — every turn, tool call, and result | Summarized session facts — what tools ran, what was found |
| **Lifetime** | One HTTP request — created at start, gone when response ends | Permanent — survives server restarts, lives in Milvus |
| **Saved by** | Nobody on the server — NestJS must persist and send it back | `VectorMemory.store_if_new()` writes to Milvus `user_memory` |
| **Read timing** | Every loop iteration — full list sent to Claude on every `ainvoke()` | Once at request start — recalled before the loop begins |
| **Write timing** | Loop appends as it runs (+2 messages per tool call) | After streaming ends — never during the loop |
| **Per user?** | Yes — client (NestJS) manages it | Yes — `user_id` filter in every Milvus query |
| **Format Claude sees** | Typed message objects in the main message list | Bullet points injected into the system prompt |
| **Grows with?** | Every tool call iteration within one request | Every session where tools were used |
| **Trimmed by?** | `TokenBudget.trim_history()` at 193,904 tokens | Never trimmed — accumulates across sessions |

### Where each lands in the Claude prompt

```text
SystemMessage         ← agent identity + rules
  └─ memory section   ← LONG-TERM memory bullet points (recalled once before loop)
HumanMessage          ← past turn  }
AIMessage             ← past turn  }  SHORT-TERM memory
HumanMessage          ← past turn  }  (NestJS sends as request.history)
AIMessage(tool_call)  ← this loop  }  (loop appends during execution)
ToolMessage(result)   ← this loop  }
HumanMessage          ← current user message
```

Long-term memory shapes **who Claude thinks it's talking to** before any reasoning starts.
Short-term memory gives Claude **the full conversation thread** to follow the dialogue turn by turn.

### The key difference in plain terms

```text
Short-term:  "What did we just say in this conversation?"
             → lives in a Python list, gone in ~1 second

Long-term:   "What do we know about this user from previous sessions?"
             → lives in Milvus, persists for weeks or months
```

### Where RAG fits alongside both

RAG is **neither** short-term nor long-term memory — it is a **knowledge retrieval tool**:

```text
Short-term memory  →  "what happened in THIS conversation"
Long-term memory   →  "what we know about THIS user across all sessions"
RAG                →  "what the company's documents say" (shared, not personal)
```

All three are injected into the same Claude prompt on every request, at different positions and moments.

---

## 7. Milvus Collections

Both memory layers use Milvus as their vector store. The shared schema is defined in `src/rag/milvus_utils.py` via `ensure_collection()`, which creates a collection if it does not exist.

### Standard schema (both collections)

```python
# src/rag/milvus_utils.py — ensure_collection()
schema.add_field("id",     DataType.INT64,         is_primary=True, auto_id=True)
schema.add_field("vector", DataType.FLOAT_VECTOR,  dim=1024)          # BGE-M3 output
# All other fields (content, source, user_id, tags…) are dynamic — no schema needed
```

### SQL analogy

```text
SQL:     database  →  table    →  row     →  column
Milvus:  .db file  →  collection → entity → field
```

### Collection comparison

| | `knowledge_base` | `user_memory` |
|---|---|---|
| **Purpose** | Company documents for RAG | Per-user long-term memory |
| **Written by** | `POST /v1/rag/ingest` | `VectorMemory.store_if_new()` |
| **Read by** | `rag_retrieve` tool in agent loop | `VectorMemory.recall()` before loop |
| **Scope** | All users share one collection | Filtered by `user_id` |
| **GDPR** | Document-level deletion | Per-user deletion required |
| **MIN_SCORE** | 0.72 | 0.78 |
| **Dynamic fields** | `content`, `source` | `content`, `user_id`, `tags`, `created_at` |

### How a vector is generated

BGE-M3 is a transformer neural network. Internally it works like this:

```text
Input text: "What is the return policy?"
        ↓
Tokenizer splits it into subwords:
  ["What", "is", "the", "return", "policy", "?"]
        ↓
Each token → a 1024-dim embedding lookup (the model's learned vocabulary table)
        ↓
12 transformer layers process relationships between all tokens
  (attention: "return" is related to "policy", not "What")
        ↓
The [CLS] token's final hidden state = the sentence representation
        ↓
Output: [0.12, -0.34, 0.88, … ] — 1024 floats  (dense_vecs)
```

The 1024 numbers don't have human-readable meaning individually. Together they encode the **direction** of the sentence's semantic meaning in a 1024-dimensional space.

### The code path (embeddings.py)

```text
caller: await embedder.embed("What is the return policy?")
        │
        ▼ src/rag/embeddings.py — EmbeddingClient.embed()
        │
        ├─ partial(_model.encode, ["What is the return policy?"],
        │          batch_size=1, max_length=512)
        │     ↑ wraps the blocking call into a zero-arg callable
        │
        ├─ await loop.run_in_executor(None, fn)
        │     ↑ offloads to a thread pool — event loop stays free
        │     ↑ BGE-M3 computes here (CPU/GPU, ~50-200ms)
        │
        └─ result["dense_vecs"][0].tolist()
              ↑ result has 3 outputs — we only use dense_vecs:
                  dense_vecs:      (1, 1024) — standard semantic vector  ← used
                  lexical_weights: token importance scores               ← ignored
                  colbert_vecs:    per-token vectors for re-ranking      ← ignored
              ↑ [0] takes the first row (we sent one text, got one result)
              ↑ .tolist() converts numpy array → Python list[float]
                          (Milvus requires list[float], not numpy)
```

### Why `run_in_executor`?

BGE-M3's `.encode()` is **synchronous** — it blocks the thread for ~50-200ms. In FastAPI's async event loop, blocking the thread would freeze the whole server. `run_in_executor` moves the work to a separate thread so other requests continue being served while the model computes.

```text
FastAPI event loop thread:
  Request A → await embed() → [suspends, free to do other work]
                                   ↓
                             Thread pool: BGE-M3 runs (~100ms)
                                   ↓
  Request A → [resumes] → gets [0.12, -0.34, 0.88, …]
```

### Where embed() is called

The same `EmbeddingClient().embed(text)` is called in four places:

| Caller | Text being embedded | Purpose |
|---|---|---|
| `routers/rag.py` ingest | each document chunk | Store in `knowledge_base` |
| `rag/retriever.py` retrieve | user's query | Search `knowledge_base` |
| `memory/vector_memory.py` recall | user's current message | Search `user_memory` |
| `memory/vector_memory.py` store_if_new | session summary text | Store in `user_memory` |

The same model, same 1024 dimensions, same cosine space — that's what makes similarity scores comparable across all four uses.

### How vector search works

```text
Query: "What is the return policy?"
        ↓
BGE-M3 locally embeds it → [0.12, -0.34, 0.88, … ] (1024 floats)
        ↓
Milvus COSINE ANN search
  → finds the TOP_K vectors "closest in angle" to query vector
  → cosine similarity score: 0.0 (unrelated) → 1.0 (identical)
        ↓
Code filters: score >= MIN_SCORE
        ↓
Returns only high-confidence matches
```

### Why COSINE similarity works

```text
"What is the return policy?"  → vector A = [0.12, -0.34, 0.88, …]
"How do I get a refund?"      → vector B = [0.10, -0.30, 0.91, …]  ← similar direction
"Today's weather is sunny."   → vector C = [-0.72, 0.03, 0.65, …]  ← different direction

COSINE(A, B) = cos(angle between A and B) ≈ 0.97  ← high similarity
COSINE(A, C) = cos(angle between A and C) ≈ 0.21  ← low similarity
```

COSINE ignores vector magnitude — it only measures the **angle** between vectors, which represents semantic direction. That's why a short tweet and a long paragraph about the same topic can have a high COSINE score. This is why COSINE is preferred over L2 (Euclidean distance) for text embeddings.

---

## 7. HTTP Routes

Three route files register all endpoints. They connect at Milvus as the shared data store.

### All endpoints

```text
app-ai (FastAPI)
│
├─ GET  /health                    → main.py        (liveness check for load balancers)
├─ GET  /info                      → main.py        (active model + config)
│
├─ POST /v1/agent/chat             → routers/agent.py  (the full AI agent)
│
├─ POST /v1/rag/ingest             → routers/rag.py    (upload doc → job_id)
└─ GET  /v1/rag/ingest/{job_id}    → routers/rag.py    (poll ingestion status)
```

### Workflow A — Document Ingestion (before users chat)

```text
NestJS
  │
  ├─ POST /v1/rag/ingest  (multipart file upload)
  │     ↓ returns {job_id, status:"queued"} immediately (< 100ms)
  │
  │   [Background task runs after response is sent]
  │     chunk_text(doc)               split into paragraphs
  │     embedder.embed(chunk)         BGE-M3 locally per chunk
  │     MilvusClient.insert()     →   "knowledge_base" collection
  │     _jobs[job_id] = "done"
  │
  └─ GET /v1/rag/ingest/{job_id}   (poll every few seconds)
        → reads _jobs dict (in-memory)
        → returns {status: "processing" | "done" | "error"}
        → NestJS stops polling when "done"
```

Why two endpoints? Document ingestion is slow (embedding every chunk requires a local model call). Returning a `job_id` immediately and polling avoids HTTP timeouts.

### Workflow B — Agent Chat (the main path)

```text
NestJS
  │
  └─ POST /v1/agent/chat  {message, user_id, history, stream}
        │
        ├─ [1] Guardrails (sync, < 1ms)
        │       content policy → PII redaction → injection detection
        │       blocked → HTTP 422 immediately, nothing else runs
        │       passed  → request.message replaced with sanitized version
        │
        ├─ [2] VectorMemory.recall(user_id, message)
        │       searches Milvus "user_memory" filtered by user_id
        │       returns list of relevant memory strings
        │
        ├─ [3] build_system_prompt(recalled_memory)
        │       assembles system prompt with memory injected
        │
        ├─ [4] build_messages(system_prompt, history)
        │       converts request.history dicts → LangChain typed objects
        │       appends current HumanMessage at the end
        │
        └─ [5] ReAct loop (up to 10 iterations)
                │
                ├─ planner.ainvoke(messages)  → Claude decides: tool or final answer?
                │
                ├─ if tool_calls → rag_retrieve:
                │     RAGRetriever.retrieve(query)
                │       searches Milvus "knowledge_base"
                │     result appended as ToolMessage
                │
                ├─ if tool_calls → other tools:
                │     tool.ainvoke(args)
                │     result appended as ToolMessage
                │
                └─ if no tool_calls → stream final answer
                        stream=True  → SSE  data: {type:"token", content:"..."}\n\n
                        stream=False → JSON {type:"complete", content:"..."}

        After streaming finishes:
        └─ VectorMemory.store_if_new()
              writes session tool summary to "user_memory"
              only if tool calls happened
              silently skipped if Milvus is down
```

### The connection between the two workflows

```text
POST /v1/rag/ingest  ──writes──▶  Milvus "knowledge_base"
                                          │
                                          ▼
POST /v1/agent/chat  ──reads──▶  rag_retrieve tool (inside agent loop)
```

The routes never call each other. Milvus is the only shared state between them.

### The `stream` flag

The same agent loop runs in both modes. Only the delivery format differs:

| `stream: true` | `stream: false` |
|---|---|
| `StreamingResponse` with SSE events | `JSONResponse` with full body |
| `data: {type:"token", content:"Hello"}\n\n` per token | `{type:"complete", content:"Hello world"}` once |
| Real-time display in UI | Background jobs / curl / testing |

---

## 8. Complete Request Lifecycle

```text
LLM prompt composition at each planner call:

  Position 1  →  SystemMessage         "You are a helpful AI assistant..."
  Position 2  →  [Long-term Memories]  "- User prefers bullet points"  ← from VectorMemory
  Position 3  →  Conversation History  HumanMessage / AIMessage turns  ← from request.history
  Position 4  →  [AIMessage]           tool_calls=[{name: rag_retrieve}] ← during loop
  Position 5  →  [ToolMessage]         "[Document 1 — source: policy.pdf]..." ← RAG result
  Position 6  →  HumanMessage          current user message

Position matters. Long-term memory is placed early so it shapes how Claude interprets
everything that follows. RAG documents and tool results are placed late so the most
recent observations are closest to Claude's "attention".
```

---

## 9. Guardrails

Every message passes through three checks before the agent loop starts. Guardrails run synchronously (no `await`) — they are regex-based and complete in under 1ms.

```text
User message
    │
    ▼
Check 1: Content Policy     — blocks violence, illegal instructions, harmful content
    │ fail → HTTP 422 immediately
    │ pass ↓
Check 2: PII Redaction      — replaces emails → [EMAIL], phones → [PHONE], etc.
    │                          never blocks — always continues with sanitized text
    │ always ↓
Check 3: Injection Detection — detects "ignore all previous instructions" attacks
    │ fail → HTTP 422 immediately
    │ pass ↓
Agent loop starts with sanitized message
```

Guardrails are the only mandatory gate. If they fail, nothing else runs — not memory recall, not the agent loop, not Milvus. This guarantees Claude never sees PII or injection attempts.

Source: `src/guardrails/checker.py` — `GuardrailChecker.check()`

---

## 10. Memory vs RAG — The Critical Distinction

Both use Milvus and BGE-M3 embeddings, but they serve fundamentally different purposes and are injected at different points in the prompt.

| Dimension | Long-term Memory | RAG |
|---|---|---|
| **Owner** | One user | Shared across all users |
| **Content** | User preferences, past task summaries | Company documents, product manuals |
| **Written by** | Agent (after each session automatically) | Ingestion pipeline (`POST /v1/rag/ingest`) |
| **Milvus collection** | `user_memory` | `knowledge_base` |
| **Recall threshold** | 0.78 — must be confident | 0.72 — broader retrieval acceptable |
| **Injected as** | System prompt section | ToolMessage mid-loop |
| **Injected when** | Before loop starts | When Claude calls `rag_retrieve` |
| **Purpose** | Personalization and continuity | Factual grounding for answers |
| **GDPR scope** | Per-user deletion required | Document-level deletion |

```text
RAG answers:     "What does our refund policy say?"
                  → retrieved from shared knowledge_base

Memory answers:  "What did this user tell us about their preferences?"
                  → retrieved from user_memory filtered by user_id

Used together in one request:

  system prompt  = identity + rules
               +  [long-term memory]   ← what we know about THIS user
  message list  = conversation history
               +  [RAG tool result]    ← facts retrieved for THIS question
               +  current user message
```

---

## 11. Memory Failure Modes

Long-term memory and RAG are **optional enrichment**, never required dependencies. The agent loop must complete correctly even when Milvus is unavailable.

| Failure | Behaviour | User impact |
|---|---|---|
| Milvus unreachable at recall | `recall()` catches exception, returns `[]` | No personalization; agent still answers |
| Milvus unreachable at write | `store_if_new()` catches exception, `pass` | Session completed; memory not saved |
| Recall returns zero results | Empty list; system prompt has no memory section | Normal for first-time users |
| Score below threshold | Chunk discarded | Prevents low-confidence facts from misleading Claude |
| RAG collection missing | `ensure_collection()` creates it on first call | Transparent to caller |
| Token budget exhausted | `trim_history()` drops oldest turns | Agent yields "partial answer" notice and exits loop |
| Guardrails block message | HTTP 422 returned | No agent loop, no Milvus access |

---

## 12. Memory Design Rules

1. **Long-term memory is written after the response is sent**, never during the agent loop. Writing during the loop would add Milvus latency to the user-visible token stream.

2. **Long-term memory is recalled once per request**, before the planner runs. Recalling mid-loop adds latency and is unnecessary — one recall at the start covers the whole session.

3. **Memory and RAG use separate Milvus collections.** `user_memory` is per-user and GDPR-scoped. `knowledge_base` is shared and not subject to per-user deletion.

4. **Guardrails run before memory recall.** If the message is blocked, no Milvus access happens at all — not memory, not RAG.

5. **The LLM does not write to memory directly.** Only `store_if_new()` called from `agent_loop.py` writes to memory after streaming ends. The agent loop, not the LLM, decides what gets stored (the tool call summary).

6. **`ensure_collection()` is the single place the schema is defined.** `src/rag/milvus_utils.py` is called by all three callers (`retriever.py`, `rag.py`, `vector_memory.py`). Schema changes need only one edit.

---

## 13. Files

```
app-ai/
└── src/
    ├── main.py                      FastAPI app, CORS, route mounting, /health, /info
    ├── routers/
    │   ├── agent.py                 POST /v1/agent/chat — guardrails + SSE/JSON dispatch
    │   └── rag.py                   POST /v1/rag/ingest, GET /v1/rag/ingest/{job_id}
    ├── agent/
    │   ├── agent_loop.py            ReAct loop — the core reasoning engine
    │   └── token_budget.py          Context window tracking and history trimming
    ├── llm/
    │   └── claude_client.py         ChatAnthropic singleton, build_messages, build_system_prompt
    ├── memory/
    │   └── vector_memory.py         Long-term per-user memory: recall() + store_if_new()
    ├── rag/
    │   ├── retriever.py             RAGRetriever: retrieve() + format_for_prompt()
    │   ├── milvus_utils.py          ensure_collection() — shared schema/index creation
    │   ├── embeddings.py            EmbeddingClient wrapping BGE-M3 (local, 1024-dim)
    │   └── chunker.py               chunk_text() — splits documents into paragraphs
    ├── guardrails/
    │   └── checker.py               GuardrailChecker — content policy, PII, injection
    ├── tools/
    │   └── registry.py              TOOLS list + tool_map for agent loop dispatch
    ├── schemas/
    │   └── request.py               AgentRequest Pydantic model
    └── config/
        └── settings.py              pydantic-settings reads .env automatically
```

Cross-references:
- [app-ai-architecture.md](app-ai-architecture.md) — full system architecture and component interaction

---

## 14. NestJS — How to Handle Short-term History

The Python agent is **stateless** — it stores nothing between requests. NestJS owns the history. The contract is:

```text
NestJS sends history IN  →  Python uses it  →  NestJS stores the new turns OUT
```

### The data contract (from `src/schemas/request.py`)

```typescript
interface HistoryMessage {
  role: "user" | "assistant"   // only these two values — "system" is rejected with 422
  content: string
}

interface AgentRequest {
  user_id:      string          // for long-term memory scoping in Milvus
  session_id:   string          // for logging and correlation
  message:      string          // current user message (1–10,000 chars)
  history:      HistoryMessage[] // prior turns — send [] on first message
  user_context: {
    subscription_tier: string
    locale:    string            // default "en-US"
    timezone:  string            // default "UTC"
  }
  stream: boolean                // default true
}
```

### NestJS responsibility: store and grow the history array

```typescript
// Conceptual NestJS ChatService

@Injectable()
export class ChatService {
  constructor(private readonly db: PrismaService) {}

  async chat(userId: string, sessionId: string, userMessage: string) {

    // ── Step 1: Load existing history for this session ─────────────────────
    const history = await this.db.message.findMany({
      where:   { sessionId },
      orderBy: { createdAt: "asc" },
      select:  { role: true, content: true },
    })
    // history = [
    //   { role: "user",      content: "What is 15% of $340?" },
    //   { role: "assistant", content: "15% of $340 is $51." },
    // ]

    // ── Step 2: Call the Python agent with full history ────────────────────
    const response = await this.callPythonAgent({
      user_id:    userId,
      session_id: sessionId,
      message:    userMessage,     // the new user turn — NOT included in history
      history,                     // all prior turns, oldest first
      user_context: {
        subscription_tier: "pro",
        locale:   "en-US",
        timezone: "America/New_York",
      },
      stream: true,
    })

    // ── Step 3: Save both the new user turn and agent answer ───────────────
    await this.db.message.createMany({
      data: [
        { sessionId, role: "user",      content: userMessage },
        { sessionId, role: "assistant", content: response.fullText },
      ],
    })

    return response
  }
}
```

### NestJS responsibility: consume the SSE stream

```typescript
// Python sends when stream: true:
//   data: {"type":"token",    "content":"Hello"}\n\n
//   data: {"type":"token",    "content":" world"}\n\n
//   data: {"type":"done"}\n\n

async callPythonAgent(body: AgentRequest): Promise<{ fullText: string }> {
  const response = await fetch("http://localhost:8000/v1/agent/chat", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  })

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let fullText = ""
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n\n")
    buffer = lines.pop()!            // keep incomplete trailing chunk

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue
      const event = JSON.parse(line.slice(6))

      if (event.type === "token") {
        fullText += event.content
        // → forward token to browser via WebSocket or NestJS SSE
      }
      if (event.type === "done") break
    }
  }

  return { fullText }
}
```

### Database schema NestJS needs

```sql
Table: sessions
  id          UUID       PRIMARY KEY
  user_id     UUID       REFERENCES users(id)
  created_at  TIMESTAMP

Table: messages
  id          UUID       PRIMARY KEY
  session_id  UUID       REFERENCES sessions(id)
  role        ENUM       ('user', 'assistant')
  content     TEXT
  created_at  TIMESTAMP
```

### The full round-trip per message

```text
Browser: "Compare with Ford"
        │
        ▼ NestJS ChatService
① SELECT * FROM messages WHERE session_id = X ORDER BY created_at ASC
  → history = [{role:"user","What is Tesla..."},{role:"assistant","Tesla revenue..."}]

② POST /v1/agent/chat
  { message: "Compare with Ford", history: [...2 turns...], ... }

③ Python assembles messages list:
  [SystemMessage, HumanMessage("Tesla?"), AIMessage("Tesla..."),
   HumanMessage("Compare with Ford")]   ← short-term memory complete
  Runs ReAct loop → streams tokens

④ NestJS collects tokens → fullText = "Ford's revenue was $18.5B..."

⑤ INSERT INTO messages:
     (session_id=X, role='user',      content='Compare with Ford')
     (session_id=X, role='assistant', content="Ford's revenue was $18.5B...")

⑥ Next request: history now has 4 turns instead of 2
```

### Key rules for NestJS

| Rule | Reason |
|---|---|
| Always send full history, oldest first | Python appends current message at the end — order matters |
| Save BOTH turns after each response | If only the user turn is saved, Claude loses its own reply next time |
| Save after streaming completes, not before | `fullText` is only complete after `type:"done"` is received |
| Use `session_id` to scope history | Different conversations must not mix turns |
| Send `history: []` on the first message | Python handles empty history gracefully — no crash |
| Do NOT include current message inside `history` | Python appends it separately via `messages.append(HumanMessage(...))` |

---

## 15. Why Short-term and Long-term Memory Are Saved in Different Services

It comes down to **what the data is for** and **who needs to own it**.

### Short-term history lives in NestJS because it is a conversation artifact

The conversation history is fundamentally a **user-facing product feature** — it's the chat thread the user sees in the UI. That means:

- NestJS already owns the user, the session, and the UI — the history naturally belongs next to those things
- The browser needs to display past messages, so NestJS must store them anyway for the UI to work, regardless of the AI agent
- If Python stored it, NestJS would still need its own copy for the UI — you'd end up with two stores of the same data

The Python agent is an **internal microservice** — it processes one request and returns a result. It doesn't know about users, sessions, or UI state. Keeping it stateless means it stays simple, horizontally scalable, and replaceable. You could swap out the Python agent entirely without touching the conversation database.

```text
NestJS owns:          users, sessions, messages, billing, auth
Python owns:          LLM reasoning, tools, embeddings, RAG
Shared boundary:      one HTTP request with history IN, answer OUT
```

### Long-term memory lives in Python (Milvus) because it is an AI reasoning artifact

Long-term memory is not a chat transcript — it is **semantic context** that only the AI can produce and only the AI knows how to use. It requires:

- **Vector embeddings** — turning text into 1024-dim floats so similarity search works
- **Cosine similarity search** — finding relevant memories by semantic meaning, not exact match
- **Threshold filtering** — discarding low-confidence matches before they reach the prompt

NestJS has no embedding model and no vector database. Even if you stored the summaries in NestJS's SQL database, you couldn't do semantic retrieval — you'd get exact-match keyword search at best, which defeats the purpose.

The Python agent is the only service that knows:
- What to store (a tool-call summary, not the raw transcript)
- How to embed it (BGE-M3)
- How to retrieve it semantically (Milvus COSINE search)
- Where to inject it (system prompt, before the loop)

```text
NestJS SQL:   "SELECT * WHERE session_id = X ORDER BY created_at"   ← exact lookup by ID
Milvus:       "search WHERE vector ≈ query_vector AND user_id = X"  ← semantic lookup by meaning
```

### The separation is a clean responsibility boundary

```text
┌─────────────────────────────────────────────────────┐
│  NestJS — Product layer                             │
│                                                     │
│  "What did the user say?"                           │
│  → stored as rows in SQL, retrieved by session_id   │
│  → needed by UI to render the chat thread           │
│  → needed by Python to give Claude context          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Python — AI layer                                  │
│                                                     │
│  "What do we know about this user's intent?"        │
│  → stored as vectors in Milvus, retrieved by cosine │
│  → only meaningful to the LLM reasoning system      │
│  → NestJS has no use for it and can't query it      │
└─────────────────────────────────────────────────────┘
```

If you moved long-term memory into NestJS, you'd have to move the embedding model, the vector database, the similarity threshold logic, and the prompt injection logic with it — you'd be rebuilding the Python agent inside NestJS for one feature.

If you moved short-term history into Python, you'd need Python to have a database per user and session management, and the history would be unavailable to the UI — NestJS would still need its own copy anyway.

Each service stores exactly what it needs to do its job, and nothing more.

---

## 16. user_memory vs knowledge_memory vs history — Full Comparison

This section answers three questions that come up when working with the memory system:

1. Why store `user_memory` at all?
2. What is the difference between `user_memory` and `history` managed by NestJS?
3. What is the difference between `user_memory` and `knowledge_memory` (RAG)?

---

### Why store user_memory?

Without `user_memory`, Claude starts every session from a blank slate — even if this user has had 50 conversations before.

```text
WITHOUT user_memory:

  Session 1:  "I'm a web developer learning AI agents"
              Claude: "Great! Here's an intro to AI agents..."
                                                             ↓ nothing saved
  Session 2 (next week):  "How do I add memory to an agent?"
              Claude: "Sure! Here's a beginner overview..."
                       ← no idea this user already knows the basics
```

```text
WITH user_memory:

  Session 1:  "I'm a web developer learning AI agents"
              Claude: answers...
              → VectorMemory.store_if_new() saves:
                "User is a web developer learning AI agents with Python"
                                                             ↓ saved to Milvus

  Session 2 (next week):  "How do I add memory to an agent?"
              → VectorMemory.recall() finds the saved fact
              → system prompt now includes:
                "## What you know about this user
                 - User is a web developer learning AI agents with Python"
              Claude: "Since you're already building in Python, here's how
                       to add Milvus-backed long-term memory to your ReAct loop..."
                       ← personalized from the start, no re-introduction needed
```

`user_memory` is what turns a stateless API call into a persistent, personalized agent.

---

### user_memory vs history (NestJS)

Both are about remembering what was said. The difference is *what kind of information* and *how long it needs to live*.

| | NestJS history | user_memory (Milvus) |
|---|---|---|
| **What it stores** | Every message, in exact order | Only durable personal facts |
| **Purpose** | Replay the conversation thread so Claude can follow "it" / "that" / "as I said" | Personalize future sessions with known user context |

| **Lifespan** | One session (or until the user deletes the thread) | Permanent across all sessions |
| **Queried by** | Exact ID and order — `SELECT WHERE session_id = X ORDER BY created_at` | Semantic similarity — `search WHERE vector ≈ query AND user_id = X` |
| **Storage** | SQL rows in NestJS database | Vector embeddings in Milvus |
| **Who writes it** | NestJS, after every turn | Python agent, after each session |
| **Who reads it** | NestJS sends it as `request.history`; Python converts it to typed messages | Python calls `recall()` before the loop starts |
| **Where it lands in the prompt** | Main message list — `HumanMessage`, `AIMessage` turns | System prompt `## What you know about this user` section |
| **Example content** | `"What is 15% of $340?"` → `"It is $51."` → `"Now add 8% tax."` | `"User is a web developer learning AI agents"` |

#### The key difference in plain terms

```text
history:     "What did we say IN THIS conversation?"
             → Needed to follow a multi-turn dialogue ("it", "that", "as I said before")
             → Lives in NestJS SQL, gone when the session is over
             → Exact sequence matters — oldest first, current message last

user_memory: "What do we know about THIS USER from previous sessions?"
             → Needed for personalisation across weeks or months
             → Lives in Milvus, persists permanently
             → Sequence doesn't matter — retrieved by semantic meaning
```

#### Why the same message is not stored in both

Your message `"I'm a web developer learning AI agents"` is already saved to NestJS history for the current session. The question for `user_memory` is different: is this fact worth injecting into the system prompt in a *future* session?

NestJS history answers: *what was said*.
`user_memory` answers: *what should Claude know about this person*.

These are different questions that require different storage strategies.

---

### user_memory vs knowledge_memory (RAG)

Both use the exact same technical stack (BGE-M3 embeddings + Milvus cosine search). The difference is *whose data it is* and *when it's written*.

| | user_memory | knowledge_memory (RAG) |
|---|---|---|
| **Content** | Facts about one specific user | Company documents, manuals, policies |
| **Scope** | Private — filtered by `user_id` on every search | Shared — all users search the same collection |
| **Written by** | Python agent automatically, after each session | Admin pipeline — `POST /v1/rag/ingest` |
| **Written when** | After the ReAct loop finishes streaming | When an admin uploads a document |
| **Milvus collection** | `user_memory` | `knowledge_base` |
| **Similarity threshold** | 0.78 — must be confident | 0.72 — broader retrieval acceptable |
| **Injected as** | System prompt `## What you know about this user` | `ToolMessage` mid-loop (when Claude calls `rag_retrieve`) |
| **Injected when** | Before the loop starts | Mid-loop, only when Claude decides to use it |
| **GDPR** | Per-user deletion required | Document-level deletion |
| **Example** | `"User prefers Python and works in finance"` | `"[Document 1 — source: refund-policy.pdf] ...content..."` |

#### Why the thresholds are different (0.78 vs 0.72)

A wrong memory fact injected into the *system prompt* shapes how Claude interprets the entire conversation before any reasoning starts. A wrong RAG chunk arrives as a *tool result* that Claude can read, reason about, and discard.

```text
Wrong memory in system prompt:
  "User is a beginner at Python"
  → Claude oversimplifies every answer for the whole session
  → User gets condescending responses they can't override

Wrong RAG chunk in ToolMessage:
  "The refund policy states 14 days..."  (actually 30 days)
  → Claude reads it but can cross-reference with other chunks
  → User can ask a follow-up and Claude corrects itself
```

A higher recall threshold (0.78) for `user_memory` means fewer memories come through, but each one is more likely to be correct. It's better to retrieve nothing than the wrong personal fact.

#### In plain terms

```text
knowledge_memory:  "What does the company's documentation say?"
                   → shared by all users, written by admins
                   → retrieved mid-conversation when Claude needs facts

user_memory:       "What do we know about this specific person?"
                   → private per user, written automatically by the agent
                   → injected at session start to personalize the whole interaction
```

---

### What gets stored in user_memory — and what doesn't

Not every user message is worth remembering. The agent uses a dedicated LLM call (`_extract_memory()` in `agent_loop.py`) to extract only durable personal facts.

```text
WORTH storing:
  "I'm a web developer learning AI agents with Python"
    → stored as: "User is a web developer learning AI agents with Python"

  "I prefer concise bullet-point answers"
    → stored as: "User prefers concise bullet-point answers"

  "I work in finance and need compliance-aware responses"
    → stored as: "User works in finance and needs compliance-aware responses"

NOT worth storing (Claude returns NONE):
  "What's the weather today?"   → question, no personal fact
  "Thanks, that was helpful!"   → acknowledgement, no personal fact
  "Can you repeat that?"        → clarification request, no personal fact
```

The `_extract_memory()` function in `agent_loop.py` sends the user's message to Claude with a focused single-turn prompt. Claude either rewrites it as a fact starting with "User" or returns "NONE". This prevents noise from accumulating in `user_memory` and keeps recalled chunks relevant.

#### Why not store every message?

```text
If every message were stored:

  user_memory after 10 messages:
  - "What's the weather today?"
  - "Thanks"
  - "Can you repeat that?"
  - "How do I add memory to an agent?"
  - "OK"
  - ...

  recall() returns up to 5 of these for the next session.
  The system prompt now contains "User said OK" and "User asked about weather".
  Those tokens wasted = fewer tokens for actual personal context.
```

```text
With _extract_memory():

  user_memory after 10 messages:
  - "User is a web developer learning AI agents with Python"
  - "User prefers concise bullet-point answers"

  recall() returns both of these.
  The system prompt section is small, accurate, and genuinely useful.
```

---

### All three layers in one prompt

Every request assembles all three layers into one Claude prompt:

```text
SystemMessage         ← agent identity + rules
  └─ memory section   ← user_memory recalled facts  (Layer 3 — long-term)
HumanMessage          ← past turn  ┐
AIMessage             ← past turn  │ NestJS history  (Layer 1 — short-term)
HumanMessage          ← past turn  ┘
AIMessage(tool_call)  ← this loop ─┐
ToolMessage(result)   ← this loop  │ knowledge_memory RAG result  (Layer 2)
HumanMessage          ← current user message
```

Each layer answers a different question:

| Layer | Question answered |
|---|---|
| `user_memory` | Who is this person and what do they care about? |
| `history` | What have we been talking about in this conversation? |
| `knowledge_memory` | What do our company documents say about this topic? |

All three are needed for a production-quality agent. Remove any one and the agent becomes noticeably worse:

- No `user_memory` → cold, impersonal responses on every new session
- No `history` → can't follow multi-turn dialogue ("what does *it* mean?")
- No `knowledge_memory` → can't answer questions about private documents
