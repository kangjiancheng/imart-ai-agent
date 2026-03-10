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

## Project Setup Development

### Prerequisites

Make sure these are installed on your machine before starting:

- **pyenv** — Python version manager (`brew install pyenv` on macOS)
- **Docker** — required if using Docker Milvus (Option B in step 6)
- **~4GB free disk space** — BGE-M3 model weights (~2.3GB) are downloaded on first run and cached at `~/.cache/modelscope/hub/models/BAAI/bge-m3/`
- **~2GB free RAM** — BGE-M3 loads into memory at server startup (uses ~1GB with fp16)

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

### 5. Download the BGE-M3 embedding model

BGE-M3 (~2.3GB) powers the vector search. Download it **before** starting the server. The current implementation uses **ModelScope SDK** (Option 1) — download once, loads from local cache on every server start.

**Option 1 — ModelScope SDK (current implementation, confirmed working):**

```bash
pip install modelscope   # already in requirements.txt
```

```python
# Download once — run this script or just start the server (it downloads automatically)
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-m3')
print(f'Cached at: {model_dir}')
# → ~/.cache/modelscope/hub/models/BAAI/bge-m3
```

Or use the CLI:

```bash
modelscope download --model BAAI/bge-m3
```

After the first download, `snapshot_download()` in `src/rag/embeddings.py` returns the local cache path instantly — no network calls on subsequent server starts.

**Option 2 — Ollama (alternative, no Python model deps):**

_Local machine:_

```bash
brew install ollama        # macOS
# Ollama starts automatically — if you see "address already in use" it is already running
ollama pull bge-m3

# Verify
curl http://localhost:11434/api/embeddings \
  -d '{"model": "bge-m3", "prompt": "test"}'
# → {"embedding": [...]}
```

_Docker Compose_ — use the combined `docker-compose.yml` in Step 6 below, which starts Ollama and Milvus together.

> **Ollama note:** Dense vectors only, input truncated to 4,096 tokens. Switching to Ollama requires updating `src/rag/embeddings.py` — see [docs/agent-rag-embeddings.md](../docs/agent-rag-embeddings.md) for the Ollama variant code.

**Option 3 — HuggingFace** (if `huggingface.co` is accessible):

```bash
python -c "
from FlagEmbedding import BGEM3FlagModel
BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print('Done.')
"
```

> If `huggingface.co` is blocked or slow, use Option 1 (ModelScope) instead. See [docs/agent-rag-embeddings.md](../docs/agent-rag-embeddings.md) for full troubleshooting.


### 6. Set up environment variables

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

| Variable            | Required | Description                                                |
| ------------------- | -------- | ---------------------------------------------------------- |
| `ANTHROPIC_API_KEY` | Yes      | Claude API key from console.anthropic.com                  |
| `OPENAI_API_KEY`    | Yes      | OpenAI key for embeddings (text-embedding-3-small)         |
| `MILVUS_URI`        | No       | Milvus URI (default: `./milvus_local.db` — Milvus Lite)    |
| `MILVUS_TOKEN`      | No       | Milvus auth token `user:password` (only for Docker Milvus) |
| `TAVILY_API_KEY`    | No       | Tavily search API key — web search is disabled if missing  |

### 6. Start infrastructure (Milvus + Ollama)

**Option A — Milvus Lite (local dev, no Docker needed)**

No setup required. Set in `.env`:

```
MILVUS_URI=./milvus_local.db
```

A local `.db` file is created automatically on first use. No Ollama needed for this option — use the FlagEmbedding or HuggingFace download options in Step 5.

**Option B — Docker Compose (Milvus + Ollama together)**

Create a `docker-compose.yml` in the `app-ai/` directory:

```yaml
services:
  milvus:
    image: milvusdb/milvus:v2.4.0
    command: milvus run standalone
    environment:
      ETCD_USE_EMBED: "true"
      ETCD_DATA_DIR: /var/lib/milvus/etcd
      COMMON_STORAGETYPE: local
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama # BGE-M3 model persists across restarts

volumes:
  milvus_data:
  ollama_data:
```

```bash
# Start both services
docker compose up -d

# Pull BGE-M3 into the Ollama container (run once — ~2.3GB)
docker exec -it ollama ollama pull bge-m3
```

Then set in `.env`:

```
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
```

```bash
# Stop
docker compose down

# Stop and remove all data
docker compose down -v
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

**Option A — Milvus Lite (default)**

```bash
cd app-ai
source venv/bin/activate
uvicorn src.main:app --reload --port 8000
```

**Option B — Docker Milvus**

```bash
bash standalone_embed.sh start   # start Milvus (enter Mac login password)
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

## Dependencies

`requirements.txt` was generated with `pip freeze`, which dumps every installed package — including packages that were pulled in automatically. Here is a breakdown of what was **manually installed** vs what came along for the ride.

### What you install manually

These are the only packages you need to `pip install` directly. Everything else in `requirements.txt` is a transitive dependency pulled in by one of these.

| Package                    | Version pinned | What it does                                                                        |
| -------------------------- | -------------- | ----------------------------------------------------------------------------------- |
| `fastapi`                  | 0.135.1        | Web framework — defines HTTP routes, validates request bodies, generates Swagger UI |
| `uvicorn`                  | 0.41.0         | ASGI server — runs the FastAPI app (`uvicorn src.main:app`)                         |
| `pydantic-settings`        | 2.13.1         | Reads `.env` file into typed Python settings (`BaseSettings`)                       |
| `langchain-anthropic`      | 1.3.4          | `ChatAnthropic` — connects to Claude with tool-calling and token streaming          |
| `langchain-core`           | 1.2.17         | LangChain message types (`HumanMessage`, `ToolMessage`) and `@tool` decorator       |
| `langchain-text-splitters` | 1.1.1          | `RecursiveCharacterTextSplitter` — splits documents into chunks for RAG             |
| `FlagEmbedding`            | 1.3.5          | Loads the local **BGE-M3** model — converts text to 1024-dim vectors for Milvus     |
| `pymilvus`                 | 2.6.9          | Milvus Python client — stores and searches vector embeddings                        |
| `milvus-lite`              | 2.5.1          | Embedded Milvus — lets pymilvus run from a local `.db` file (no Docker needed)      |
| `tavily-python`            | 0.7.22         | Tavily web search API client — powers the `web_search` tool                         |
| `python-multipart`         | 0.0.22         | Enables file uploads in FastAPI (`UploadFile`) — used by `/v1/rag/ingest`           |
| `pytest`                   | —              | Test runner — used for `tests/` unit tests                                          |

### What gets pulled in automatically (transitive dependencies)

You never need to `pip install` these. They appear in `requirements.txt` because `pip freeze` includes the full dependency tree.

| Package                          | Pulled in by                             | Why it's needed                                     |
| -------------------------------- | ---------------------------------------- | --------------------------------------------------- |
| `anthropic`                      | `langchain-anthropic`                    | The actual Anthropic HTTP client under the hood     |
| `httpx`, `httpcore`              | `anthropic`                              | HTTP client used by the Anthropic SDK               |
| `starlette`                      | `fastapi`                                | FastAPI is built on top of Starlette                |
| `pydantic`, `pydantic-core`      | `fastapi`, `pydantic-settings`           | Data validation engine                              |
| `python-dotenv`                  | `pydantic-settings`                      | Reads the `.env` file                               |
| `torch`                          | `FlagEmbedding`                          | PyTorch — BGE-M3 runs on top of it (~2 GB download) |
| `transformers`                   | `FlagEmbedding`                          | HuggingFace model loading                           |
| `sentence-transformers`          | `FlagEmbedding`                          | Sentence embedding utilities                        |
| `huggingface-hub`                | `transformers`                           | Downloads model weights from HuggingFace cache      |
| `accelerate`                     | `transformers`                           | Optimized model loading on CPU/GPU                  |
| `numpy`, `scipy`, `scikit-learn` | `FlagEmbedding`, `sentence-transformers` | Math / ML utilities used internally                 |
| `datasets`, `tokenizers`         | `FlagEmbedding`                          | HuggingFace data + tokenisation libraries           |
| `tiktoken`                       | `langchain-anthropic`                    | Token counting for context window management        |
| `langsmith`                      | `langchain-core`                         | LangChain tracing and observability (optional)      |
| `tenacity`                       | `langchain-core`                         | Retry logic for API calls                           |
| `grpcio`, `protobuf`             | `pymilvus`                               | gRPC transport used by the Milvus client            |
| `anyio`, `sniffio`               | `fastapi`, `httpx`                       | Async I/O compatibility layer                       |
| `uvloop`                         | `uvicorn`                                | Fast event loop backend on macOS/Linux              |
| `aiohttp`, `yarl`                | various                                  | Async HTTP used internally                          |

### Why the list looks so large

`FlagEmbedding` is the main reason — it pulls in the entire PyTorch + HuggingFace ecosystem (`torch`, `transformers`, `sentence-transformers`, `accelerate`, `datasets`, `safetensors`, `tokenizers`, etc.). That single package is responsible for most of the ~80 entries you see in `requirements.txt`.

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
