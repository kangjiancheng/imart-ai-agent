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

| Method | Path                      | Description                                                             |
| ------ | ------------------------- | ----------------------------------------------------------------------- |
| `POST` | `/v1/agent/chat`          | Main streaming endpoint. Returns SSE token stream.                      |
| `POST` | `/v1/rag/ingest`          | Upload a document to the knowledge base. Returns `job_id`.              |
| `GET`  | `/v1/rag/ingest/{job_id}` | Poll ingestion job status (`queued` / `processing` / `done` / `error`). |
| `GET`  | `/health`                 | Health check for load balancers and deployment probes.                  |
| `GET`  | `/docs`                   | Interactive Swagger UI (auto-generated).                                |

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

### Testing the agent chat endpoint

**Option 1 — Swagger UI (easiest)**

Visit `http://127.0.0.1:8000/docs` in your browser → click `POST /v1/agent/chat` → click **Try it out**.

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

| Step | File                    | What it does                                     | On fail                       |
| ---- | ----------------------- | ------------------------------------------------ | ----------------------------- |
| 1    | `content_policy.py`     | Blocks harmful requests (weapons, hacking, etc.) | 422 error                     |
| 2    | `pii_filter.py`         | Redacts email, phone, SSN, credit card numbers   | Never blocks — cleans message |
| 3    | `injection_detector.py` | Blocks prompt injection attempts                 | 422 error                     |

---

## RAG knowledge base

Documents are ingested via `POST /v1/rag/ingest` and stored in the Milvus `knowledge_base` collection. At query time, the agent embeds the user's message and retrieves the most semantically similar chunks (score ≥ 0.72).

**Ingestion flow:**

```text
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

## Project Setup

### Prerequisites

Make sure these are installed on your machine before starting:

- **pyenv** — Python version manager (`brew install pyenv` on macOS)
- **Docker** — required to run Milvus (vector database)

### 1. Enter the project folder

```bash
cd app-ai
```

### 2. Install and pin the Python version

**Option A**, install by the pyenv, like the nvm in the web node.js switch the node version.

```bash
pyenv install 3.11.9    # skip if already installed
pyenv local 3.11.9      # writes .python-version — commit this file
python --version        # → Python 3.11.9
```

**Option B**, install by the brew directly, then update the path in the shell profile:

```bash
brew install python@3.12
```

then, update the shell profile

```bash
# Add this to ~/.zshrc
echo 'export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

python --version    # → Python 3.12.13
```

### 3. Create and activate a virtual environment

A virtual environment is like `node_modules` — it keeps this project's packages isolated from everything else on your machine.

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux — prompt changes to (venv)
# venv\Scripts\activate       # Windows
```

exit the virtual environment:

```bash
deactivate
```

> **Always activate venv before running `pip` or `python`.** If you forget, packages install globally and projects conflict.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up environment variables

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

| Variable            | Required | Description                                               |
| ------------------- | -------- | --------------------------------------------------------- |
| `ANTHROPIC_API_KEY` | Yes      | Claude API key from console.anthropic.com                 |
| `OPENAI_API_KEY`    | Yes      | OpenAI key for embeddings (text-embedding-3-small)        |
| `MILVUS_HOST`       | No       | Milvus host (default: `localhost`)                        |
| `MILVUS_PORT`       | No       | Milvus port (default: `19530`)                            |
| `TAVILY_API_KEY`    | No       | Tavily search API key — web search is disabled if missing |

### 6. Start Milvus (vector database, via Docker)

```bash
docker run -d --name milvus \
  -p 19530:19530 \
  milvusdb/milvus:latest standalone
```

### 7. Run the server

```bash
uvicorn src.main:app --reload --port 8000
```

Server starts at `http://localhost:8000`.

| URL                                        | What it does                                    |
| ------------------------------------------ | ----------------------------------------------- |
| `http://127.0.0.1:8000/docs`               | Interactive Swagger UI — try all endpoints here |
| `http://127.0.0.1:8000/redoc`              | ReDoc API reference                             |
| `http://127.0.0.1:8000/health`             | Returns `{"status": "ok", "service": "app-ai"}` |
| `POST http://127.0.0.1:8000/v1/agent/chat` | Main agent endpoint                             |
| `POST http://127.0.0.1:8000/v1/rag/ingest` | RAG ingest endpoint                             |

> Start with `/docs` — it gives you a full interactive UI to test the API directly from the browser without needing Postman or curl.

---

### Daily workflow (after first-time setup)

```bash
cd app-ai
source venv/bin/activate
uvicorn src.main:app --reload --port 8000
```

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
