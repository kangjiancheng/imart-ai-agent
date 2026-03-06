# app-ai

AI agent microservice for the app platform. Built with FastAPI and LangChain, powered by Claude (Anthropic). Exposes a streaming SSE endpoint that NestJS calls to handle user AI requests.

---

## What it does

- Runs a **ReAct agent loop** (Reason → Act → Observe → repeat) to answer user questions
- Streams response tokens word-by-word to the client via **Server-Sent Events (SSE)**
- Uses **RAG** (Retrieval-Augmented Generation) to answer questions from private documents stored in Milvus
- Maintains **per-user long-term memory** across sessions using vector embeddings
- Enforces a **guardrail pipeline** on every message before Claude sees anything

---

## Architecture

```
NestJS
  │
  └─► POST /v1/agent/chat
        │
        ├─ 1. Guardrails (content policy → PII redaction → injection detection)
        │
        ├─ 2. Memory recall (Milvus: user_memory collection)
        │
        ├─ 3. ReAct loop
        │       ├─ Claude decides: answer directly OR call a tool
        │       ├─ Tool: rag_retrieve → Milvus similarity search
        │       ├─ Tool: web_search  → Tavily API
        │       └─ Tool: calculator  → safe AST-based math eval
        │
        └─ 4. Stream final answer tokens → SSE → NestJS
              + store session summary to Milvus memory
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/agent/chat` | Main streaming endpoint. Returns SSE token stream. |
| `POST` | `/v1/rag/ingest` | Upload a document to the knowledge base. Returns `job_id`. |
| `GET` | `/v1/rag/ingest/{job_id}` | Poll ingestion job status (`queued` / `processing` / `done` / `error`). |
| `GET` | `/health` | Health check for load balancers and deployment probes. |
| `GET` | `/docs` | Interactive Swagger UI (auto-generated). |

### Request body — `POST /v1/agent/chat`

```json
{
  "user_id": "user-123",
  "session_id": "session-abc",
  "message": "What is the refund policy?",
  "history": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help?" }
  ],
  "user_context": {
    "subscription_tier": "pro",
    "locale": "en-US",
    "timezone": "America/New_York"
  }
}
```

### Response — SSE stream

```
data: The refund policy
data:  allows returns within
data:  30 days of purchase.
data: [DONE]
```

---

## Project structure

```
app-ai/
├── src/
│   ├── main.py                   # FastAPI app entry point, middleware, routers
│   ├── config/
│   │   └── settings.py           # pydantic-settings: reads .env, validates config
│   ├── schemas/
│   │   ├── request.py            # AgentRequest, HistoryMessage, UserContext
│   │   └── response.py           # AgentResponse, Document
│   ├── guardrails/
│   │   ├── checker.py            # Orchestrates all 3 checks; GuardrailResult
│   │   ├── content_policy.py     # Regex: blocks harmful requests
│   │   ├── pii_filter.py         # Regex: redacts email, phone, SSN, card numbers
│   │   └── injection_detector.py # Regex: blocks prompt injection attempts
│   ├── llm/
│   │   └── claude_client.py      # ChatAnthropic singleton; builds messages + system prompt
│   ├── agent/
│   │   ├── agent_loop.py         # Core ReAct loop — async generator, streams tokens
│   │   └── token_budget.py       # Tracks context window; trims history when near limit
│   ├── tools/
│   │   ├── registry.py           # TOOLS list + tool_map used by the agent loop
│   │   ├── web_search.py         # Tavily web search tool
│   │   ├── calculator.py         # Safe AST-based math evaluator (no eval())
│   │   ├── database.py           # Database query tool stub
│   │   └── code_exec.py          # Code execution tool stub
│   ├── rag/
│   │   ├── embeddings.py         # OpenAI text-embedding-3-small via AsyncOpenAI
│   │   ├── retriever.py          # Milvus similarity search; format_for_prompt()
│   │   ├── chunker.py            # RecursiveCharacterTextSplitter wrapper
│   │   └── reranker.py           # Optional Cohere cross-encoder reranker stub
│   ├── memory/
│   │   └── vector_memory.py      # Per-user long-term memory: recall() + store_if_new()
│   └── routers/
│       ├── agent.py              # POST /v1/agent/chat — guardrails + StreamingResponse
│       └── rag.py                # POST/GET /v1/rag/ingest — background document ingestion
└── tests/
    └── unit/
        ├── test_guardrails.py    # Tests for content policy, PII, injection checks
        └── test_token_budget.py  # Tests for TokenBudget constants and trimming
```

---

## How files work together

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INCOMING REQUEST                                │
│                    POST /v1/agent/chat (JSON)                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/main.py                                                            │
│  "The front door"                                                       │
│  Creates the FastAPI app, registers middleware (CORS), mounts routers   │
│  Every HTTP request enters here first                                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/schemas/request.py                                                 │
│  "The bouncer at the door"                                              │
│  Defines what valid JSON looks like (AgentRequest, UserContext, etc.)   │
│  FastAPI auto-validates — wrong shape = 422 before your code runs       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  valid request object
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/routers/agent.py                                                   │
│  "The traffic cop"                                                      │
│  Receives the validated request, decides what happens next              │
│  Routes: /v1/agent/chat → run guardrails → start streaming             │
└──────────────┬──────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  src/guardrails/                     │
│  "The security checkpoint"           │
│                                      │
│  checker.py        ← orchestrator    │
│    │                                 │
│    ├─ content_policy.py              │
│    │  blocks: "how to make a bomb"   │
│    │                                 │
│    ├─ pii_filter.py                  │
│    │  redacts: email, phone, SSN     │
│    │  (never blocks, only cleans)    │
│    │                                 │
│    └─ injection_detector.py          │
│       blocks: "ignore all rules"     │
│                                      │
│  All regex. No LLM. < 1ms.           │
│  Blocked → 422 error, done.          │
│  Passed → sanitized message          │
└──────────────┬───────────────────────┘
               │  clean message
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  src/agent/agent_loop.py                                                 │
│  "THE BRAIN — the core of everything"                                    │
│                                                                          │
│  1. recall memory      ──→  src/memory/vector_memory.py                  │
│                             "What do we know about this user?"           │
│                                                                          │
│  2. build messages     ──→  src/llm/claude_client.py                     │
│                             converts history dicts → LangChain objects   │
│                             injects system prompt + recalled memory      │
│                                                                          │
│  3. ReAct loop (repeats up to 10x):                                      │
│     ┌──────────────────────────────────────────────────────┐             │
│     │  ask Claude: what should I do?                       │             │
│     │       │                                              │             │
│     │       ├── "I'll answer directly" ──→ exit loop       │             │
│     │       │                                              │             │
│     │       └── "I need a tool"                            │             │
│     │               │                                      │             │
│     │               ├─ rag_retrieve ──→ src/rag/           │             │
│     │               │                  retriever.py        │             │
│     │               │                  embeddings.py       │             │
│     │               │                                      │             │
│     │               ├─ web_search  ──→ src/tools/          │             │
│     │               │                  web_search.py       │             │
│     │               │                                      │             │
│     │               └─ calculator  ──→ src/tools/          │             │
│     │                                  calculator.py       │             │
│     │                                                      │             │
│     │  tool result → back to Claude → reason again → ...   │             │
│     └──────────────────────────────────────────────────────┘             │
│                                                                          │
│  4. stream final answer  token by token → SSE → NestJS                  │
│                                                                          │
│  5. store memory       ──→  src/memory/vector_memory.py                  │
│                             "remember this session"                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why each directory has one job

```
src/
├── config/       ← env vars only. No logic.
├── schemas/      ← data shapes only. No logic.
├── guardrails/   ← security only. Runs before any AI.
├── llm/          ← Claude connection only. One singleton.
├── agent/        ← the loop brain. Coordinates everything.
├── tools/        ← things Claude can DO (search, calculate)
├── rag/          ← knowledge retrieval (read documents)
├── memory/       ← user memory (remember people)
└── routers/      ← HTTP endpoints only. Thin wrappers.
```

Dependency direction always flows inward — `routers` calls `agent_loop`, `agent_loop` calls `tools/rag/memory`, nothing calls back up. This means you can swap any one piece (e.g. change the embedding model in `rag/`) without touching anything else.

---

## Guardrail pipeline

Every message passes through three checks **before** Claude sees anything. All checks are regex-based — no LLM call, < 1ms total.

| Step | File | What it does | On fail |
|------|------|-------------|---------|
| 1 | `content_policy.py` | Blocks harmful requests (weapons, hacking, etc.) | 422 error |
| 2 | `pii_filter.py` | Redacts email, phone, SSN, credit card numbers | Never blocks — cleans message |
| 3 | `injection_detector.py` | Blocks prompt injection attempts | 422 error |

---

## RAG knowledge base

Documents are ingested via `POST /v1/rag/ingest` and stored in the Milvus `knowledge_base` collection. At query time, the agent embeds the user's message and retrieves the most semantically similar chunks (score ≥ 0.72).

**Ingestion flow:**
```
File upload → decode UTF-8 → chunk (RecursiveCharacterTextSplitter)
  → embed each chunk (OpenAI) → store in Milvus knowledge_base
```

---

## Long-term memory

Per-user facts are stored in the Milvus `user_memory` collection, filtered by `user_id`. Memory is recalled before the loop starts and written after streaming ends.

- Recall threshold: **0.78** (higher than RAG — must be confident before injecting into prompt)
- Write: session summary stored with tags after every completed session
- Graceful degradation: Milvus unavailable → returns `[]` → agent continues without memory

---

## Setup

**1. Copy and fill in environment variables**

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key from console.anthropic.com |
| `OPENAI_API_KEY` | Yes | OpenAI key for embeddings (text-embedding-3-small) |
| `MILVUS_HOST` | No | Milvus host (default: `localhost`) |
| `MILVUS_PORT` | No | Milvus port (default: `19530`) |
| `TAVILY_API_KEY` | No | Tavily search API key — web search is disabled if missing |

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Start Milvus (via Docker)**

```bash
docker run -d --name milvus \
  -p 19530:19530 \
  milvusdb/milvus:latest standalone
```

**4. Run the server**

```bash
uvicorn src.main:app --reload --port 8000
```

Server starts at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

---

## Running tests

```bash
pytest tests/
```

---

## Key design decisions

**Why LangChain's `ChatAnthropic` instead of the raw Anthropic SDK?**
`ChatAnthropic` provides `bind_tools()` for structured tool calling and `astream()` for token streaming. It also makes a future upgrade to LangGraph straightforward without migrating code.

**Why not use `eval()` in the calculator?**
`eval()` executes arbitrary Python. Since Claude constructs the expression string from user input, a crafted message could trigger `eval("__import__('os').system('rm -rf /')")`. The AST-based evaluator only allows numbers and safe math operators.

**Why two separate Milvus collections?**
`knowledge_base` is shared company documents (no user filter). `user_memory` is personal per-user facts (always filtered by `user_id`). Mixing them would let one user's memories appear in another's context.

**Why return error strings from tools instead of raising exceptions?**
Tool exceptions would crash the agent loop and return a 500 to the user. Returning an error string lets Claude read the failure and either try another tool or tell the user gracefully.
