# app-ai-base

Production-ready LangChain ReAct AI agent powered by Claude. This is the clean base version of `app-ai` — all tutorial and Python-learning comments have been removed, leaving only functional docstrings and architectural notes.

## Stack

| Layer           | Technology                                                |
| --------------- | --------------------------------------------------------- |
| Framework       | FastAPI + uvicorn                                         |
| LLM             | Claude (via `langchain-anthropic`)                        |
| Embeddings      | BAAI/bge-m3 — local, free, multilingual (~1 GB)           |
| Vector DB       | Milvus (Docker) or Milvus Lite (file-based, no Docker)    |
| Reranker        | BAAI/bge-reranker-v2-m3 — local, optional Cohere fallback |
| Package Manager | [uv](https://github.com/astral-sh/uv)                     |

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Docker (if using full Milvus; not needed for Milvus Lite)
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Install dependencies

```bash
cd app-ai-base
uv sync
```

For dev dependencies (pytest):

```bash
uv sync --extra dev
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
ANTHROPIC_API_KEY=sk-ant-xxx
MILVUS_URI=http://localhost:19530   # or ./milvus_local.db for Milvus Lite
MILVUS_TOKEN=root:Milvus            # leave empty for Milvus Lite
```

### 3. Start Milvus (Docker, skip if using Milvus Lite)

```bash
# Download and start standalone Milvus
curl -O https://raw.githubusercontent.com/milvus-io/milvus/master/configs/milvus.yaml
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest standalone
```

Or use the included compose file from `app-ai/`:

```bash
docker compose -f ../app-ai/milvus-standalone-docker-compose.yml up -d
```

### 4. Run the server

```bash
uv run uvicorn src.main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health: [http://localhost:8000/health](http://localhost:8000/health)

### 5. Send a test request

```bash
curl -X POST http://localhost:8000/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-001",
    "session_id": "session-001",
    "message": "What is 15% tip on a $47.80 meal?",
    "user_context": { "subscription_tier": "free" },
    "stream": false
  }'
```

---

## API Reference

### POST `/v1/agent/chat`

Chat with the AI agent. Supports SSE streaming (default) or single JSON response.

**Request body:**

```json
{
  "user_id": "base-user-001",
  "session_id": "001",
  "message": "What is AI agent Skill?",
  "history": [
    { "role": "user", "content": "Hello!" },
    { "role": "assistant", "content": "What can I do for you?" }
  ],
  "user_context": {
    "subscription_tier": "free | pro | enterprise",
    "locale": "en-US",
    "timezone": "UTC"
  },
  "stream": true
}
```

**SSE events (stream=true):**

```
data: {"type": "token",    "content": "Hello"}
data: {"type": "token",    "content": " world"}
data: {"type": "done"}
```

**Single response (stream=false):**

```json
{ "type": "complete", "content": "Hello world" }
```

---

### POST `/v1/agent/chat-with-file`

Multipart upload: ask a question about a document (PDF, DOCX, TXT).

```bash
curl -X POST http://localhost:8000/v1/agent/chat-with-file \
  -F "file=@contract.pdf" \
  -F "message=Summarize clause 5" \
  -F "user_id=user-001" \
  -F "session_id=session-001" \
  -F "subscription_tier=pro"
```

---

### POST `/v1/rag/ingest`

Ingest a document into the persistent knowledge base (chunked + embedded into Milvus).

```bash
curl -X POST http://localhost:8000/v1/rag/ingest \
  -F "file=@product-manual.pdf"
# → {"job_id": "550e8400-...", "status": "queued"}
```

### GET `/v1/rag/ingest/{job_id}`

Poll ingestion status: `queued` → `processing` → `done` | `error`.

### GET `/v1/rag/chunks?source=filename.pdf`

Inspect stored chunks for a source file (verify extraction quality).

### GET `/v1/rag/content?source=filename.pdf`

Return the full reconstructed text for a source file.

### DELETE `/v1/rag/chunks?source=filename.pdf`

Delete all chunks for a source file (use before re-ingesting a corrected file).

---

## Project Structure

```
app-ai-base/
├── src/
│   ├── main.py                  # FastAPI app entry point — CORS, routers, /health, /info
│   │
│   ├── agent/
│   │   ├── agent_loop.py        # ReAct loop — yields SSE tokens; runs tools; writes memory
│   │   └── token_budget.py      # Context window tracker — trims history at 193,904 tokens
│   │
│   ├── config/
│   │   └── settings.py          # pydantic-settings — reads .env, validates required keys
│   │
│   ├── guardrails/
│   │   ├── checker.py           # Orchestrates 3 checks in sequence
│   │   ├── content_policy.py    # Check 1 — regex blocklist for harmful requests
│   │   ├── pii_filter.py        # Check 2 — replaces email/phone/SSN/card with tokens
│   │   └── injection_detector.py # Check 3 — detects prompt-override attack phrases
│   │
│   ├── llm/
│   │   └── claude_client.py     # ChatAnthropic singleton; build_messages; build_system_prompt
│   │
│   ├── memory/
│   │   └── vector_memory.py     # Per-user long-term memory — recall() and store_if_new()
│   │
│   ├── rag/
│   │   ├── retriever.py         # RAGRetriever — vector search + rerank → list[Document]
│   │   ├── embeddings.py        # EmbeddingClient — BGE-M3 local model, async via thread pool
│   │   ├── milvus_utils.py      # ensure_collection() — shared schema/index setup helper
│   │   ├── chunker.py           # chunk_text() and chunk_documents() via LangChain splitter
│   │   └── reranker.py          # rerank() — BGE reranker → Cohere → original order fallback
│   │
│   ├── routers/
│   │   ├── agent.py             # POST /v1/agent/chat and /v1/agent/chat-with-file
│   │   └── rag.py               # POST/GET /v1/rag/ingest, GET/DELETE /v1/rag/chunks|content
│   │
│   ├── schemas/
│   │   ├── request.py           # AgentRequest, HistoryMessage, UserContext (Pydantic)
│   │   └── response.py          # AgentResponse, Document (Pydantic, for OpenAPI docs)
│   │
│   ├── tools/
│   │   ├── registry.py          # TOOLS list (for llm.bind_tools) + tool_map dict (for execution)
│   │   ├── web_search.py        # TavilySearch tool (stub if TAVILY_API_KEY not set)
│   │   ├── calculator.py        # AST-based safe math evaluator (no eval())
│   │   ├── code_exec.py         # Stub — sandboxed code execution (disabled by default)
│   │   └── database.py          # Stub — read-only SQL query tool (disabled by default)
│   │
│   └── utils/
│       └── file_parser.py       # extract_text() — PDF/DOCX/TXT → plain string, capped 80k chars
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── pyproject.toml               # uv project definition + dependencies
├── .env.example                 # Environment variable template
└── README.md
```

---

## Architecture Overview

### ReAct Agent Loop (`agent/agent_loop.py`)

```
User message
    │
    ├─ recall()           VectorMemory → Milvus user_memory → inject into system prompt
    │
    ├─ build_messages()   Assemble [SystemMessage, ...history, HumanMessage]
    │
    └─ ReAct loop (max 10 iterations)
           │
           ├─ planner.ainvoke()   Full response to check tool_calls
           │        │
           │        ├─ tool_calls present?
           │        │      ├─ rag_retrieve → RAGRetriever.retrieve() → Milvus search
           │        │      └─ other tools  → tool_map[name].ainvoke(args)
           │        │      append AIMessage + ToolMessages → next iteration
           │        │
           │        └─ no tool_calls → planner.astream() → yield tokens → break
           │
           └─ store_if_new()   Extract personal fact + session summary → Milvus user_memory
```

### Memory vs RAG

|            | VectorMemory                            | RAGRetriever                    |
| ---------- | --------------------------------------- | ------------------------------- |
| Collection | `user_memory`                           | `knowledge_base`                |
| Scope      | Per user (filtered by `user_id`)        | Shared across all users         |
| MIN_SCORE  | 0.40                                    | 0.50                            |
| Purpose    | Personal facts across sessions          | Company docs, policies, manuals |
| Written by | `_extract_memory()` after each response | `/v1/rag/ingest` endpoint       |

### Guardrail Pipeline

```
User message → content_policy → pii_filter → injection_detector → Claude
                   (block)         (scrub)           (block)
```

All checks are regex-based (< 1 ms). Runs synchronously before any LLM call.

### SSE Streaming

```
POST /v1/agent/chat
    → StreamingResponse(token_stream(), media_type="text/event-stream")
        → async for token in run(request): yield "data: {...}\n\n"
        → yield "data: {type: done}\n\n"
```

Headers `Cache-Control: no-cache` and `X-Accel-Buffering: no` prevent proxy buffering.

---

## Environment Variables

| Variable                 | Required | Default             | Description                                     |
| ------------------------ | -------- | ------------------- | ----------------------------------------------- |
| `ANTHROPIC_API_KEY`      | ✅       | —                   | Anthropic API key                               |
| `MILVUS_URI`             | ✅       | —                   | `http://localhost:19530` or `./milvus_local.db` |
| `MILVUS_TOKEN`           |          | `""`                | `root:Milvus` for Docker Milvus; empty for Lite |
| `ANTHROPIC_BASE_URL`     |          | `None`              | Custom proxy or on-prem endpoint                |
| `TAVILY_API_KEY`         |          | `None`              | Tavily web search API key                       |
| `CLAUDE_MODEL`           |          | `claude-sonnet-4-6` | Claude model ID                                 |
| `CLAUDE_MAX_TOKENS`      |          | `4096`              | Max response tokens                             |
| `RAG_TOP_K`              |          | `5`                 | Max RAG results per query                       |
| `RAG_MIN_SCORE`          |          | `0.50`              | Minimum cosine similarity to include a chunk    |
| `AGENT_MAX_ITERATIONS`   |          | `10`                | Max ReAct loop iterations per request           |
| `TOKENIZERS_PARALLELISM` |          | —                   | Set `false` to suppress HuggingFace warning     |

---

## Running Tests

```bash
uv run pytest
```

---

## Adding a New Tool

1. Create `src/tools/my_tool.py` with a `@tool` decorated function and a clear docstring.
2. Import it in `src/tools/registry.py` and add it to both `TOOLS` and `tool_map`.
3. Claude will automatically discover it via the docstring on the next restart.

```python
# src/tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """What this tool does. Claude reads this to decide when to use it."""
    return "result"
```

```python
# src/tools/registry.py
from src.tools.my_tool import my_tool

TOOLS    = [web_search, calculator, rag_retrieve, my_tool]
tool_map = {t.name: t for t in TOOLS if t.name != "rag_retrieve"}
```
