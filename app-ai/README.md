# app-ai agent

AI agent microservice for the app platform. Built with FastAPI and LangChain, powered by Claude (Anthropic). Exposes a streaming SSE endpoint that NestJS calls to handle user AI requests.

This installs FastAPI, LangChain, Milvus (via docker), BGE-M3 (FlagEmbedding via ModelScope), PyMuPDF, TAVILY.

### What it does

- Runs a **ReAct agent loop** (Reason → Act → Observe → repeat) to answer user questions
- Streams response tokens word-by-word to the client via **Server-Sent Events (SSE)**
- Accepts **file uploads alongside a message** — extracts text from PDF, DOCX, or TXT and injects it into context
- Uses **RAG** (Retrieval-Augmented Generation) to answer questions from private documents stored in Milvus
- Maintains **per-user long-term memory** across sessions using vector embeddings
- Enforces a **guardrail pipeline** on every message before Claude sees anything

## Daily workflow

```bash
# Start Docker Milvus if set the env configure for milvus server
bash standalone_embed.sh start     # requires sudo password

cd app-ai

 # Set Up a Virtual Environment
python -m venv venv

 # stop: deactivate
source venv/bin/activate

# Install Required Dependencies
pip install -r requirements.txt

uvicorn src.main:app --reload --port 8000
```

Then open: http://localhost:8000/docs

> First start takes ~30–60 seconds while BGE-M3 loads into RAM. Subsequent starts are faster.

| URL                          | What it does                                  |
| ---------------------------- | --------------------------------------------- |
| http://localhost:8000/docs   | Swagger UI — test all endpoints interactively |
| http://localhost:8000/redoc  | ReDoc API reference                           |
| http://localhost:8000/health | `{"status": "ok", "service": "app-ai"}`       |
| http://localhost:8000/info   | Service config (model, max_tokens, embedding) |

## Quick start development guide

### Prerequisites

| Tool            | Required    | Purpose                                                             |
| --------------- | ----------- | ------------------------------------------------------------------- |
| Python 3.12+    | Yes         | Runtime                                                             |
| pip             | Yes         | Package manager                                                     |
| pyenv (macOS)   | Recommended | Python version isolation — like nvm for Node                        |
| Docker          | Optional    | Only needed for Docker Milvus (Option B below)                      |
| ~4 GB free disk | Yes         | BGE-M3 model weights (~2.3 GB) downloaded on first start            |
| ~1 GB free RAM  | Yes         | BGE-M3 runs with fp16 — loads at server startup (~30–60s first run) |

### 1. Install Python

**Option A — pyenv (recommended)**

```bash
brew install pyenv                 # macOS
pyenv install 3.12.13
pyenv local 3.12.13                # writes .python-version — commit this
python --version                   # → Python 3.12.13
```

**Option B — Homebrew direct**

```bash
brew install python@3.12
# if python --version error, run the following command
# echo 'export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"' >> ~/.zshrc
# source ~/.zshrc
python --version                   # → Python 3.12.x
```

### 2. Set Virtual environment

```bash
cd app-ai
python -m venv venv
source venv/bin/activate           # prompt changes to (venv)
# deactivate                       # exit venv when done
```

> Always activate venv before running `pip` or `python`. Without it, packages install globally and projects conflict.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

First install takes several minutes — PyTorch alone is ~2 GB.

### 4. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in your values
```

```ini
# Required
ANTHROPIC_BASE_URL=http://xxx
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Milvus — choose one option:
MILVUS_URI=http://localhost:19530  # Option A: Docker Milvus via standalone_embed.sh (default)
MILVUS_TOKEN=root:Milvus           # Option A only
# MILVUS_URI=./milvus_local.db     # Option B: Milvus Lite (no Docker, for local quick development)

# Optional
TAVILY_API_KEY=tvly-your-key-here  # web search — disabled if missing

# Recommended — prevents tokenizer parallelism warnings
TOKENIZERS_PARALLELISM=false
```

| Variable                 | Required | Default             | Notes                                                             |
| ------------------------ | -------- | ------------------- | ----------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`      | **Yes**  | —                   | App crashes at startup if missing                                 |
| `MILVUS_URI`             | **Yes**  | —                   | `./milvus_local.db` for Lite, `http://localhost:19530` for Docker |
| `MILVUS_TOKEN`           | No       | `""`                | Set `root:Milvus` only for Docker Milvus                          |
| `TAVILY_API_KEY`         | No       | `null`              | Web search tool disabled when missing                             |
| `CLAUDE_MODEL`           | No       | `claude-sonnet-4-6` | Override to use a different Claude model                          |
| `ANTHROPIC_BASE_URL`     | No       | `null`              | Custom proxy endpoint — uses official API if not set              |
| `TOKENIZERS_PARALLELISM` | No       | `true`              | Set `false` to suppress HuggingFace tokenizer warnings            |

### 5. Milvus setup

**Option A — Docker Milvus via `standalone_embed.sh`, default**

The included `standalone_embed.sh` script manages a Milvus standalone Docker container (version `v2.6.11`). It creates `embedEtcd.yaml` and `user.yaml` config files automatically, and persists data in `./volumes/milvus/`.

```bash
# Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# First run — pulls the Milvus image and starts the container (sudo required)
bash standalone_embed.sh start
# → waits until container is healthy (~30s)
```

more base:

```bash
# Check container is running
docker ps | grep milvus-standalone

# Stop (data is preserved in ./volumes/milvus/)
bash standalone_embed.sh stop

# Restart
bash standalone_embed.sh restart

# Delete container + all data (prompts for confirmation — irreversible)
bash standalone_embed.sh delete

# Upgrade to latest Milvus version
bash standalone_embed.sh upgrade
```

After starting, update `.env`:

```ini
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
```

**Option B — Milvus Lite (no Docker)**

No setup needed. Set `MILVUS_URI=./milvus_local.db` in `.env` (already the default in `.env.example`). A local `.db` file is created automatically on first use.

## Architecture

```
NestJS
  │
  ├─► POST /v1/agent/chat            (JSON body)
  │        │
  │        ├─ 1. Guardrails       content policy → PII redaction → injection detection
  │        │
  │        ├─ 2. Memory recall    Milvus user_memory (inject past context into system prompt)
  │        │
  │        ├─ 3. ReAct loop       up to 10 iterations
  │        │       ├─ Claude decides: answer directly OR call a tool
  │        │       ├─ rag_retrieve  → Milvus knowledge_base similarity search
  │        │       ├─ web_search    → Tavily API (skipped if TAVILY_API_KEY missing)
  │        │       └─ calculator    → safe AST-based math eval (no eval())
  │        │
  │        └─ 4. Stream final answer tokens → SSE → NestJS
  │               + store session summary → Milvus user_memory
  │
  └─► POST /v1/agent/chat-with-file  (multipart form-data)
           │
           ├─ 1. Extract text from file   PDF (PyMuPDF + OCR) / DOCX / TXT
           │
           └─ 2. Same guardrails → ReAct loop → SSE stream
                  (extracted text injected into system prompt as document context)
```

## Project structure

```
app-ai/
├── src/
│   ├── main.py                        FastAPI app — creates app, CORS middleware, mounts routers, /health, /info
│   │
│   ├── config/
│   │   └── settings.py                pydantic-settings — reads .env, validates all config at startup
│   │
│   ├── schemas/
│   │   ├── request.py                 AgentRequest, HistoryMessage, UserContext — incoming JSON shapes
│   │   └── response.py                AgentResponse, Document — outgoing JSON shapes
│   │
│   ├── guardrails/
│   │   ├── checker.py                 GuardrailChecker — runs all 3 checks in sequence
│   │   ├── content_policy.py          Regex: blocks harmful requests (weapons, hacking)
│   │   ├── pii_filter.py              Regex: redacts email, phone, SSN, card numbers (never blocks)
│   │   └── injection_detector.py      Regex: blocks prompt injection attempts
│   │
│   ├── llm/
│   │   └── claude_client.py           ChatAnthropic singleton + build_messages() + build_system_prompt()
│   │
│   ├── agent/
│   │   ├── agent_loop.py              run() — the core ReAct loop, async generator, streams SSE tokens  ← KEY FILE
│   │   └── token_budget.py            TokenBudget — trims history near Claude's 200k context limit
│   │
│   ├── tools/
│   │   ├── registry.py                TOOLS list (passed to bind_tools) + tool_map dict + rag_retrieve stub
│   │   ├── web_search.py              TavilySearch factory — returns stub tool if TAVILY_API_KEY missing
│   │   ├── calculator.py              @tool: safe AST math eval (no eval(), no code injection risk)
│   │   ├── database.py                @tool stub: read-only SQL query (requires implementation)
│   │   └── code_exec.py               @tool stub: sandboxed code execution (requires security hardening)
│   │
│   ├── rag/
│   │   ├── retriever.py               RAGRetriever — Milvus similarity search + format_for_prompt()
│   │   ├── embeddings.py              EmbeddingClient — BGE-M3 via ModelScope (local, free, 1024-dim)
│   │   ├── chunker.py                 chunk_text() and chunk_documents() — RecursiveCharacterTextSplitter
│   │   ├── milvus_utils.py            ensure_collection() — shared Milvus collection + index setup
│   │   └── reranker.py                Optional Cohere cross-encoder reranker stub
│   │
│   ├── memory/
│   │   └── vector_memory.py           VectorMemory — per-user long-term memory: recall() + store_if_new()
│   │
│   ├── utils/
│   │   └── file_parser.py             extract_text() — PDF (PyMuPDF+OCR), DOCX, plain text extraction
│   │
│   └── routers/
│       ├── agent.py                   POST /v1/agent/chat + POST /v1/agent/chat-with-file
│       └── rag.py                     POST/GET /v1/rag/ingest + GET/DELETE /v1/rag/chunks + GET /v1/rag/content
│
├── tests/
│   └── unit/
│       ├── test_guardrails.py         Tests: content policy, PII redaction, injection detection
│       └── test_token_budget.py       Tests: TokenBudget constants and history trimming
│
├── standalone_embed.sh                Milvus Docker management script (start/stop/delete/upgrade)
├── requirements.txt                   pip freeze output — all installed packages with versions
├── .env.example                       Template for .env — commit this, never commit .env
└── Dockerfile                         Container image for deployment
```

## How files work together

### Full request lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INCOMING REQUEST                                │
│             POST /v1/agent/chat (JSON)                                  │
│             POST /v1/agent/chat-with-file (multipart)                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/main.py  "The front door"                                          │
│  Creates the FastAPI app, registers CORS middleware, mounts routers.    │
│  Every HTTP request enters here first.                                  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/schemas/request.py  "The bouncer"                                  │
│  FastAPI auto-validates the JSON body against AgentRequest (Pydantic).  │
│  Wrong shape or missing field → 422 error before your code ever runs.   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  validated AgentRequest object
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/routers/agent.py  "The traffic cop"                                │
│  Receives the validated request. Runs guardrails first.                 │
│  chat-with-file: calls src/utils/file_parser.py to extract text first. │
│  Blocked → 422 immediately.  Passed → start streaming.                 │
└──────────────┬──────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  src/guardrails/  "The security checkpoint"                      │
│                                                                  │
│  checker.py  ← orchestrator — runs checks 1 → 2 → 3 in order   │
│    │                                                             │
│    ├─ content_policy.py     blocks: "how to make a bomb"         │
│    │                        → 422 on match                       │
│    │                                                             │
│    ├─ pii_filter.py         redacts: email → [EMAIL]             │
│    │                        never blocks — only cleans message   │
│    │                                                             │
│    └─ injection_detector.py blocks: "ignore all instructions"    │
│                              → 422 on match                      │
│                                                                  │
│  All regex. No LLM call. Completes in < 1ms.                    │
└──────────────┬───────────────────────────────────────────────────┘
               │  sanitized message (PII replaced)
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  src/agent/agent_loop.py  "THE BRAIN"                                    │
│                                                                          │
│  1. recall memory   ──►  src/memory/vector_memory.py                     │
│                          Milvus similarity search on user_memory         │
│                          → up to 5 past facts injected into system prompt│
│                          threshold: 0.78 (higher than RAG — must be      │
│                          confident before injecting personal context)    │
│                                                                          │
│  2. build messages  ──►  src/llm/claude_client.py                        │
│                          history dicts → typed LangChain message objects │
│                          SystemMessage + HumanMessage + AIMessage chain  │
│                          system prompt = identity + rules + recalled mem │
│                          + optional document context (chat-with-file)    │
│                                                                          │
│  3. bind tools      ──►  src/tools/registry.py                           │
│                          llm.bind_tools(TOOLS) tells Claude what tools   │
│                          are available by embedding their JSON schemas   │
│                          planner = llm with tools attached               │
│                                                                          │
│  4. ReAct loop  (max 10 iterations)                                      │
│     ┌─────────────────────────────────────────────────────────┐          │
│     │  planner.ainvoke(messages)  ← full response needed      │          │
│     │       │                       to read tool_calls        │          │
│     │       │  (ainvoke waits for complete response;          │          │
│     │       │   tool_calls would be incomplete mid-stream)    │          │
│     │       │                                                 │          │
│     │       ├── no tool_calls → stream final answer → break   │          │
│     │       │       planner.astream() yields tokens one-by-one│          │
│     │       │       (uses planner not llm — tool schemas must │          │
│     │       │        be present for tool_use history blocks)  │          │
│     │       │                                                 │          │
│     │       └── tool_calls present → execute tool             │          │
│     │               │                                         │          │
│     │               ├─ rag_retrieve ──► src/rag/retriever.py  │          │
│     │               │   special case: NOT in tool_map         │          │
│     │               │   stub in registry.py = declaration     │          │
│     │               │   RAGRetriever = real async execution   │          │
│     │               │   embed query → Milvus cosine search    │          │
│     │               │   top_k=5, min_score=0.50               │          │
│     │               │   → format_for_prompt() → ToolMessage   │          │
│     │               │                                         │          │
│     │               ├─ web_search  ──► src/tools/web_search.py│          │
│     │               │   Tavily API → up to 5 text snippets    │          │
│     │               │   returns error string if key missing   │          │
│     │               │   (never crashes the loop)              │          │
│     │               │                                         │          │
│     │               └─ calculator  ──► src/tools/calculator.py│          │
│     │                   AST-safe math eval (no eval() risk)   │          │
│     │                   e.g. "340 * 0.15" → "51.0"            │          │
│     │                                                         │          │
│     │  tool result → append AIMessage + ToolMessage to list   │          │
│     │  → next iteration: Claude sees full history + result    │          │
│     │  → Claude reasons again and either calls another tool   │          │
│     │    or declares it has enough info to answer             │          │
│     └─────────────────────────────────────────────────────────┘          │
│                                                                          │
│  5. stream final answer  tokens → SSE → NestJS                          │
│                                                                          │
│  6. store memory   ──►  src/memory/vector_memory.py                      │
│                         session summary written AFTER streaming ends     │
│                         (writing during stream would add latency)        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Tool lifecycle — register → decide → execute

Claude never calls tools directly. It says "I want to call X with these args." The agent loop executes it and feeds the result back.

```
STAGE 1: REGISTER              STAGE 2: CLAUDE DECIDES         STAGE 3: EXECUTE
─────────────────              ───────────────────────         ────────────────
tools/registry.py              agent_loop.py                   agent_loop.py

TOOLS = [                      planner =                       if tool_name == "rag_retrieve":
  web_search,                    llm.bind_tools(TOOLS)           RAGRetriever.retrieve()
  calculator,                  ↓                               else:
  rag_retrieve,                response =                        tool_map[name].ainvoke()
]                                planner.ainvoke(messages)
                               ↓
tool_map = {                   tool_call =
  "web_search": ...,             response.tool_calls[0]
  "calculator": ...,           ↓
  # rag_retrieve excluded      tool_name, tool_args extracted
}
# tool_map = {t.name: t for t in TOOLS if t.name != "rag_retrieve"}
```

**How Claude picks the right tool** — it reads each tool's docstring:

| Tool           | Docstring Claude reads                            | Example question                   |
| -------------- | ------------------------------------------------- | ---------------------------------- |
| `web_search`   | "current information, news, or recent events"     | "What happened in the news today?" |
| `calculator`   | "evaluate a math expression"                      | "What is 15% tip on $340?"         |
| `rag_retrieve` | "internal knowledge base, company docs, policies" | "What does our refund policy say?" |

If no tool is needed, `response.tool_calls` is empty and Claude answers directly.

**Why `rag_retrieve` is a stub in registry.py** — the `@tool` decorator only needs the docstring and type hints to generate a JSON schema for Claude. The function body (`return ""`) is never executed. The real work happens in `rag/retriever.py` via `RAGRetriever`, which is `async` — it must be `await`-ed in the loop, which a `@tool` stub cannot do cleanly.

**Example: multi-tool conversation trace**

```
User: "What is 15% of 340, and what does our return policy say?"

Iteration 1 — Claude emits:
  tool_call { name: "calculator", args: { expression: "340 * 0.15" } }
  → agent_loop: tool_map["calculator"].ainvoke({"expression": "340 * 0.15"})
  → result: "51.0"
  → messages += [AIMessage(tool_calls=[...]), ToolMessage("51.0")]

Iteration 2 — Claude reads "51.0", now emits:
  tool_call { name: "rag_retrieve", args: { query: "return policy" } }
  → agent_loop: RAGRetriever().retrieve("return policy")
  → result: "[Document 1 — source: policy.pdf]\nReturns accepted within 30 days..."
  → messages += [AIMessage(tool_calls=[...]), ToolMessage("[Document 1...]")]

Iteration 3 — Claude reads both results, tool_calls = [] (done)
  → agent_loop: planner.astream(messages) → streams tokens one-by-one
  → "15% of 340 is 51. Our return policy allows returns within 30 days."
```

---

### Why each directory has one responsibility

```
src/
├── config/       env vars only — no logic, no imports from other src/ modules
├── schemas/      data shapes only — Pydantic models, no business logic
├── guardrails/   security only — runs before any LLM call, pure regex, < 1ms
├── llm/          Claude connection only — one singleton, message builders
├── agent/        the loop brain — coordinates all other modules, owns the ReAct flow
├── tools/        things Claude can DO — each tool is one file, one @tool function
├── rag/          knowledge retrieval — embed → search → format, no agent logic
├── memory/       user memory — recall and store, separate from shared knowledge_base
├── utils/        shared utilities — file text extraction (PDF, DOCX, TXT)
└── routers/      HTTP endpoints only — thin wrappers that call agent_loop or rag pipeline
```

Dependency direction always flows **inward**: `routers` → `agent_loop` → `tools/rag/memory` → `llm/config`. Nothing calls back up. You can swap any one layer (e.g. replace BGE-M3 with another embedding model in `rag/embeddings.py`) without touching anything above it.

---

## Guardrail pipeline

Every message passes through three checks **before** Claude sees anything. All checks are regex — no LLM call, < 1ms total.

| Step | File                    | What it does                                                        | On fail        |
| ---- | ----------------------- | ------------------------------------------------------------------- | -------------- |
| 1    | `content_policy.py`     | Blocks harmful requests (weapons, hacking, violence, etc.)          | 422 error      |
| 2    | `pii_filter.py`         | Redacts email, phone, SSN, credit card numbers in place             | Cleans message |
| 3    | `injection_detector.py` | Blocks prompt injection ("ignore all instructions", "you are now…") | 422 error      |

Step 2 never blocks — it sanitizes the message and passes it forward. Claude only ever sees the redacted version.

---

## RAG knowledge base

Documents are ingested via `POST /v1/rag/ingest` and stored in the Milvus `knowledge_base` collection.

**Ingestion flow:**

```
File upload
    │
    ├─ PDF  → PyMuPDF table extraction (find_tables)
    │         + get_text() for paragraph text (handles Chinese CID fonts)
    │         + Tesseract OCR fallback for image-based pages
    │
    ├─ DOCX → python-docx (paragraphs + tables)
    │
    └─ TXT / other → UTF-8 decode
         │
         ▼
    RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
         │
         ▼
    BGE-M3 embed each chunk (1024-dim vector)
         │
         ▼
    store {vector, content, source} in Milvus knowledge_base
```

**Retrieval flow (at query time):**

```
User question → BGE-M3 embed → Milvus cosine similarity search (top 5)
  → filter: score >= 0.50 (RAG_MIN_SCORE)
  → format_for_prompt() → ToolMessage → Claude reads and answers
```

If no chunks pass the 0.50 threshold, `format_for_prompt([])` returns `"No relevant documents found in the knowledge base."` — Claude sees this as the tool result and decides whether to try `web_search` or answer from training knowledge.

> **Why 0.50?** The default was lowered from 0.72 to accommodate OCR-extracted text, which contains noise (garbled characters, misaligned columns) that reduces cosine similarity scores. 0.50 still filters unrelated content while allowing imperfect OCR matches through. Override via `RAG_MIN_SCORE=` in `.env`.

---

### RAG fallback chain

When `rag_retrieve` finds no matching documents, the agent degrades gracefully — it never crashes:

```
rag_retrieve returns []
  │
  └─► ToolMessage: "No relevant documents found in the knowledge base."
        │
        Claude reads the result and decides next step:
        │
        ├─ calls web_search (if question needs live/external data)
        │     ├─ TAVILY_API_KEY set   → Tavily fetches web results → answer
        │     └─ TAVILY_API_KEY missing → ToolMessage: "Web search is not configured"
        │                                  → Claude answers from training knowledge
        │
        └─ answers directly from Claude training knowledge (general questions)
```

Claude never inspects the database directly. It reads the tool result string and decides what to do next — exactly like a human reading "no results found."

---

## Endpoints

| Method   | Path                       | Description                                                                  |
| -------- | -------------------------- | ---------------------------------------------------------------------------- |
| `POST`   | `/v1/agent/chat`           | Main streaming endpoint. Returns SSE token stream.                           |
| `POST`   | `/v1/agent/chat-with-file` | Upload a file alongside a message. Text extracted and injected into context. |
| `POST`   | `/v1/rag/ingest`           | Upload a document to the knowledge base. Returns `job_id`.                   |
| `GET`    | `/v1/rag/ingest/{job_id}`  | Poll ingestion job status (`queued` / `processing` / `done` / `error`).      |
| `GET`    | `/v1/rag/chunks`           | Inspect stored chunks for a source file. Query param: `source`.              |
| `DELETE` | `/v1/rag/chunks`           | Delete all chunks for a source file before re-ingesting.                     |
| `GET`    | `/v1/rag/content`          | Return full extracted text for a source file as one joined string.           |
| `GET`    | `/health`                  | Health check for load balancers and deployment probes.                       |
| `GET`    | `/info`                    | Service config (model name, max_tokens, embedding model).                    |
| `GET`    | `/docs`                    | Interactive Swagger UI (auto-generated by FastAPI).                          |

### Request body — `POST /v1/agent/chat`

```json
{
  "user_id": "user-123",
  "session_id": "session-abc",
  "message": "What is the AI Agent?",
  "history": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help?" }
  ],
  "user_context": {
    "subscription_tier": "pro",
    "locale": "en-US",
    "timezone": "America/New_York"
  },
  "stream": true
}
```

## Long-term memory

Per-user facts stored in Milvus `user_memory`, always filtered by `user_id`.

- **Recall**: runs at loop start — top 5 similar memories injected into system prompt
- **Write**: runs after streaming ends — session tool-call summary stored with `tags: ["session"]`
- **Threshold**: 0.78 (higher than RAG's 0.50 — must be confident before injecting personal context)
- **Graceful degradation**: Milvus unavailable → `recall()` returns `[]` → agent continues without personalization

`knowledge_base` and `user_memory` are kept in separate collections: shared documents have no user filter; personal memories are always scoped by `user_id`. Mixing them would leak one user's memory into another's context.

---

## Dependencies

`requirements.txt` is generated with `pip freeze` — it includes every installed package, including transitive dependencies pulled in automatically. Below is a breakdown of what was **manually installed** and what came along for the ride.

### Manually installed (direct dependencies)

| Package                    | Version | What it does                                                                     |
| -------------------------- | ------- | -------------------------------------------------------------------------------- |
| `fastapi`                  | 0.135.1 | Web framework — HTTP routes, request validation, Swagger UI generation           |
| `uvicorn`                  | 0.41.0  | ASGI server — runs the FastAPI app                                               |
| `pydantic-settings`        | 2.13.1  | Reads `.env` into typed Python settings (`BaseSettings`)                         |
| `langchain-anthropic`      | 1.3.4   | `ChatAnthropic` — Claude with `bind_tools()` and `astream()`                     |
| `langchain-core`           | 1.2.17  | Message types (`HumanMessage`, `ToolMessage`) and `@tool` decorator              |
| `langchain-text-splitters` | 1.1.1   | `RecursiveCharacterTextSplitter` — splits documents into chunks for RAG          |
| `langchain-tavily`         | 0.2.17  | `TavilySearch` — official LangChain web search tool (recommended for agents)     |
| `FlagEmbedding`            | 1.3.5   | Loads BGE-M3 — converts text to 1024-dim vectors (local, free, multilingual)     |
| `modelscope`               | 1.34.0  | Downloads and caches BGE-M3 (avoids HuggingFace network issues)                  |
| `pymilvus`                 | 2.6.9   | Milvus Python client — stores and searches vector embeddings                     |
| `milvus-lite`              | 2.5.1   | Embedded Milvus — runs from a local `.db` file, no Docker needed                 |
| `pymupdf`                  | 1.27.2  | PDF text extraction — fast, handles Chinese CID fonts, table extraction          |
| `pytesseract`              | latest  | OCR fallback for image-based PDF pages (requires `tesseract` system binary)      |
| `pillow`                   | latest  | Image processing — used by pytesseract to read page pixmaps                      |
| `python-docx`              | latest  | DOCX text and table extraction for chat-with-file endpoint                       |
| `python-multipart`         | 0.0.22  | Enables file uploads in FastAPI (`UploadFile`) — used by ingest + chat-with-file |
| `python-dotenv`            | 1.2.2   | Reads `.env` file (used internally by pydantic-settings)                         |
| `pytest`                   | —       | Test runner                                                                      |

### Key transitive dependencies (pulled in automatically)

| Package                 | Pulled in by          | What it does                                      |
| ----------------------- | --------------------- | ------------------------------------------------- |
| `anthropic`             | `langchain-anthropic` | Raw Anthropic API client                          |
| `torch`                 | `FlagEmbedding`       | PyTorch — BGE-M3 model runtime (~2 GB install)    |
| `transformers`          | `FlagEmbedding`       | HuggingFace Transformers — model loading          |
| `sentence-transformers` | `FlagEmbedding`       | Sentence embedding utilities                      |
| `langsmith`             | `langchain-core`      | LangChain observability (optional tracing)        |
| `grpcio`                | `pymilvus`            | gRPC transport for Milvus server communication    |
| `tiktoken`              | `langchain-anthropic` | Token counting for context window management      |
| `openai`                | transitive            | Present in requirements but embeddings use BGE-M3 |

> **Note on `transformers` version:** `FlagEmbedding==1.3.5` requires `transformers==4.44.2`. Do **not** upgrade `transformers` to v5.x — it removes `is_torch_fx_available` which crashes BGE-M3 at startup. This also means `unstructured` (which forces `transformers>=5.x`) cannot coexist with this project.
