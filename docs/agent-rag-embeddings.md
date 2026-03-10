# RAG Embeddings — What They Are & Which Model to Use

> **Scope:** Explains embeddings and compares model options for the RAG + memory pipeline in `app-ai`.
> For the full RAG pipeline, see [agent-rag.md](agent-rag.md).
> For the memory system, see [agent-memory.md](agent-memory.md).

---

## 1. What Is an Embedding?

An embedding is a way of converting **text into numbers** so a computer can measure meaning.

Computers cannot directly compare sentences like humans do. But they can measure the **distance between two lists of numbers**. An embedding model converts every piece of text into a fixed-length list of numbers (called a **vector**) such that texts with similar meaning produce vectors that are numerically close to each other.

```text
"What is our refund policy?"  →  [0.023, -0.417, 0.891, ..., 0.142]
"How do I get a refund?"      →  [0.019, -0.411, 0.887, ..., 0.139]  ← close! (similar meaning)
"What's the weather today?"   →  [-0.72,  0.031, 0.651, ..., -0.830] ← far away (different meaning)
```

The number of values in the list is called the **dimension**. A 1024-dimensional model produces a list of 1024 numbers per text. More dimensions can capture more nuance, but also require more memory and storage.

### Why embeddings are the foundation of RAG

The entire RAG pipeline depends on embeddings:

```text
① Ingest time (when you upload a document):
   Document text → embedding model → vector → stored in Milvus

② Query time (when a user asks a question):
   User question → embedding model → vector → Milvus finds nearby vectors → retrieves similar documents
```

The search is only as good as the embedding. A model that does not understand your language or domain will cluster unrelated content together, causing irrelevant documents to rank highly.

**Critical rule:** The embedding model used at ingest time and at query time must be **identical**. Switching models requires deleting all stored vectors and re-ingesting every document.

### Embeddings vs. the LLM

The embedding model and Claude are **completely separate**. Claude generates the response. The embedding model only powers vector search. They never talk to each other.

```text
User question
     │
     ├──→ Embedding model → Milvus search → retrieved documents ─┐
     │                                                            ▼
     └──────────────────────────────────────────────────→ Claude (LLM) → response
```

Anthropic does not offer an embedding API. That is why this project uses a separate embedding model alongside Claude.

---

## 2. Model Comparison

Four models worth knowing for this project:

|                      | `text-embedding-3-small` | `text-embedding-3-large` | `all-MiniLM-L6-v2`        | `BAAI/bge-m3`                    |
| -------------------- | ------------------------ | ------------------------ | ------------------------- | -------------------------------- |
| **Provider**         | OpenAI                   | OpenAI                   | HuggingFace               | BAAI (open-source)               |
| **Hosting**          | Cloud API                | Cloud API                | Local                     | Local                            |
| **Cost**             | $0.02 / 1M tokens        | $0.13 / 1M tokens        | Free                      | Free                             |
| **Dimensions**       | 1536                     | 3072                     | 384                       | 1024                             |
| **Max input**        | 8,191 tokens             | 8,191 tokens             | 256 tokens                | 8,192 tokens                     |
| **Multilingual**     | Partial                  | Partial                  | English only              | 100+ languages                   |
| **Model size**       | — (API)                  | — (API)                  | ~90 MB                    | ~2.3 GB                          |
| **Startup time**     | Instant                  | Instant                  | ~1s                       | ~10s                             |
| **Latency per call** | ~100ms (network)         | ~120ms (network)         | ~20–80ms (CPU)            | ~300–600ms (CPU) / ~10ms (GPU)   |
| **Best for**         | Quick start, English     | High accuracy, English   | Lightweight, English-only | Multilingual, best quality, free |

---

## 3. BGE-M3 — The Recommended Model for This Project

### What BGE-M3 is

**BGE-M3** is an open-source embedding model built by BAAI (Beijing Academy of Artificial Intelligence). The "M3" stands for:

- **Multi-Functionality** — supports three types of retrieval in one model
- **Multi-Linguality** — trained on 100+ languages including Korean, English, Chinese, Japanese
- **Multi-Granularity** — handles short sentences and long documents (up to 8,192 tokens)

It consistently ranks at or near the top of multilingual retrieval benchmarks (MTEB Multilingual).

### BGE-M3 is a functional superset of OpenAI embeddings

OpenAI's `text-embedding-3-small` does one thing: **dense retrieval**. BGE-M3 does three. Think of it as a Swiss Army knife — one tool that covers what a screwdriver, a knife, and a can opener each do separately.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         BGE-M3 output                               │
│                                                                      │
│  Dense vector   1024-dim  ← same concept as OpenAI's output         │
│  Sparse vector  ~250K-dim ← keyword matching (like Elasticsearch)   │
│  ColBERT        per-token ← token-level fine-grained matching        │
│                                                                      │
│                         OpenAI output                                │
│  Dense vector   1536-dim  ← dense only, no other modes              │
└─────────────────────────────────────────────────────────────────────┘
```

### The three retrieval modes explained

**Dense retrieval** (output: 1024-dim vector)

The standard mode — same concept as OpenAI. Converts the entire text into one fixed-length vector that captures overall semantic meaning. Similar meaning = vectors point in the same direction.

```text
Query: "iPhone manufacturer"
Match: "Apple company"  ← high similarity, even though no words overlap
Miss:  "Samsung Galaxy" ← different meaning, different vector direction
```

**Sparse retrieval** (output: ~250,000-dim sparse vector)

Produces a high-dimensional vector where most values are zero, and non-zero values correspond to important keywords in the text. Works like traditional keyword search (BM25/Elasticsearch).

```text
Query: "Python tutorial"
Match: any document containing both "Python" and "tutorial" ← exact word match
Miss:  document about "Django guide" ← similar topic but different exact words
```

Dense retrieval would find "Django guide" as relevant. Sparse retrieval would not — it only fires on exact keyword matches. Both behaviors are useful depending on the content.

**ColBERT — multi-vector retrieval** (output: one vector per token)

Instead of compressing the whole text into one vector, ColBERT produces a separate vector for every word (token). At search time it compares token-by-token between query and document. This is the most accurate mode but requires more storage.

```text
Good at: "Find the paragraph in this 50-page document where section 3.2 mentions X"
         ← needs fine-grained matching within a long document, not just overall topic
```

### Which mode does this project use?

Currently **dense only** — the same as OpenAI. Milvus Lite (the embedded file-based database used in local development) supports dense search. Sparse and ColBERT require a full Milvus server with hybrid search configured.

This means on this project, BGE-M3's advantage over OpenAI comes from **multilingual quality, privacy, and zero API cost** — not from the sparse/ColBERT modes. Those modes become available if you later upgrade to a full Milvus server.

### Why BGE-M3 fits this project

| Reason             | Detail                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **No API cost**    | Runs locally — zero cost at any scale                                                                                              |
| **Multilingual**   | If the project serves Korean or mixed-language content, BGE-M3 handles it natively. OpenAI's model is primarily English-optimized. |
| **Long documents** | Handles up to 8,192 tokens. `all-MiniLM-L6-v2` cuts off at 256 tokens, which is too short for real documents.                      |
| **Quality**        | Top-tier retrieval quality, comparable to or better than OpenAI on multilingual content                                            |
| **Privacy**        | Data never leaves your machine — important if documents are sensitive                                                              |

---

## 4. BGE-M3 vs. `text-embedding-3-small` — Key Differences

These are the two most realistic choices for this project. Here is where they differ in practice:

### Language coverage

`text-embedding-3-small` was trained primarily on English text. It can handle other languages to some extent, but its multilingual performance degrades noticeably on Korean and Chinese compared to its English quality.

`BAAI/bge-m3` was explicitly trained on 100+ languages with equal attention to non-English content. For Korean documents, it is significantly more accurate.

### Cost model

`text-embedding-3-small` charges per token. At low traffic this is cheap ($0.02 per million tokens). At high traffic or with large document ingestion pipelines, it adds up. BGE-M3 has zero per-token cost — you pay only once in compute time.

### Latency profile

```text
text-embedding-3-small:
  Every call goes to OpenAI's servers and back.
  Fast on average (~100ms), but adds network dependency.
  If OpenAI has an outage → your embeddings stop working.

BAAI/bge-m3:
  Loads ~2GB model into memory at startup (~10 seconds, once).
  After that, each call is local — no network.
  CPU: ~300–600ms per call.
  GPU: ~5–30ms per call (comparable to or faster than OpenAI).
```

### Startup cost

BGE-M3 requires a one-time model download (~2.3GB) from HuggingFace, and takes ~10 seconds to load into memory when the server starts. After that, it runs entirely in memory. OpenAI has no startup cost since it is a remote API.

### Data privacy

With `text-embedding-3-small`, every piece of text you embed is sent to OpenAI's servers. For internal documents, support tickets, or any sensitive content, this is a consideration. BGE-M3 processes everything locally — nothing leaves your machine.

### Summary

```text
Priority                      → Choose
──────────────────────────────────────────────────────────────
Fastest setup, pure semantics → text-embedding-3-small
  No infrastructure to manage, just an API key.
  Best if content is English-focused and traffic is low.

Keyword matching / hybrid     → BGE-M3
  Need to match exact product codes, IDs, or names.
  Sparse mode handles this; OpenAI cannot.

Cost and privacy              → BGE-M3
  Long-term: server cost beats per-token API fees at scale.
  Data never leaves your machine.

Multilingual / Korean         → BGE-M3
  Explicitly trained on 100+ languages with equal weight.
  OpenAI degrades on non-English content.
```

> **Note for this project:** BGE-M3's sparse and ColBERT modes require a full Milvus server. With Milvus Lite (local dev), only dense mode is active — the same as OpenAI. BGE-M3's current advantage here is multilingual quality, privacy, and zero API cost.

---

## 5. Integration Code

### Option A — `text-embedding-3-small` (current default)

```python
# src/rag/embeddings.py
#
# AsyncOpenAI = non-blocking OpenAI client.
# `await` means: pause this coroutine and let other requests run
# while waiting for OpenAI's API to respond.

from openai import AsyncOpenAI
from src.config.settings import settings

class EmbeddingClient:
    DIMENSIONS = 1536

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding  # list of 1536 floats
```

### Option B — `BAAI/bge-m3` (recommended for multilingual / production)

```python
# src/rag/embeddings.py
#
# WHY run_in_executor?
#   FlagEmbedding's encode() is synchronous — it blocks the thread until done.
#   Calling it directly inside `async def` would freeze the entire event loop,
#   meaning FastAPI could not process any other requests while encoding.
#   run_in_executor() offloads the blocking call to a separate thread,
#   so the event loop stays free. This is the standard pattern for using
#   synchronous libraries inside async Python code.
#
# WHY a module-level _model variable?
#   Loading BGE-M3 takes ~10 seconds and uses ~2GB of RAM.
#   If we created a new model inside __init__, every EmbeddingClient()
#   instantiation would reload the model — extremely slow.
#   A module-level variable is created once when the module is first
#   imported (at server startup) and reused for every request.

from FlagEmbedding import BGEM3FlagModel

# Loaded ONCE at server startup. All requests share this single instance.
# use_fp16=True: use 16-bit floats instead of 32-bit → half the memory, minimal quality loss.
_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
```

Install:

```bash
pip install FlagEmbedding
```

#### What downloads when

`pip install FlagEmbedding` installs only the **Python library code** (a few MB, fast). The 2.3GB model weights are **not** downloaded here.

The model downloads the **first time your code loads it** — at the `BGEM3FlagModel(...)` line when the server starts:

```text
pip install FlagEmbedding     ← library code only (~MB, fast, no model yet)

uvicorn src.main:app          ← server starts
  → imports embeddings.py
  → hits BGEM3FlagModel(...)  ← HERE: downloads 2.3GB from HuggingFace
                                        (first run only, then cached forever)
  → server ready
```

After the first download, the model is cached permanently on disk. Every subsequent server start reads from cache (~10 seconds to load into RAM — no network needed).

```text
Cache location:
  macOS / Linux:  ~/.cache/huggingface/hub/models--BAAI--bge-m3/
  Windows:        C:\Users\<you>\.cache\huggingface\hub\...
```

#### When it re-downloads

| Situation                             | Downloads again?            |
| ------------------------------------- | --------------------------- |
| Server restart                        | No — cache persists on disk |
| New developer on their own machine    | Yes — first time only       |
| Docker container with no volume mount | Yes — every deploy          |
| Docker with a mounted cache volume    | No                          |
| You delete `~/.cache/huggingface/`    | Yes                         |

#### Docker: pre-bake the model into the image

If your Docker container starts fresh each deploy, it would re-download 2.3GB every time. Fix: download the model at **image build time** so it is baked in:

```dockerfile
FROM python:3.11-slim

RUN pip install FlagEmbedding

# Download model at BUILD time → baked into the image layer.
# No network download at container startup.
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)"
```

#### If HuggingFace is unreachable (timeout / blocked)

**Option 1 — Download via ModelScope (avoids HuggingFace entirely):**

```bash
pip install modelscope
```

Two ways to download — both produce the same result:

```bash
# CLI (progress bar, resume on failure)
modelscope download --model BAAI/bge-m3

# 下载单个文件到指定本地文件夹（以下载README.md到当前路径下“bge-m3-model”目录为例）
# modelscope download --model BAAI/bge-m3 README.md --local_dir ./bge-m3-model
```

Model is cached at: `~/.cache/modelscope/hub/models/BAAI/bge-m3`

```python
# Python SDK
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-m3')
print(model_dir)  # → ~/.cache/modelscope/hub/models/BAAI/bge-m3

# or 将模型下载到当前目录下的 'bge-m3-model' 文件夹
# model_dir = snapshot_download('BAAI/bge-m3', cache_dir='./bge-m3-model')
```

```python
# src/rag/embeddings.py — ModelScope variant (current implementation)
#
# snapshot_download("BAAI/bge-m3"):
#   - First call: downloads ~2.3GB from ModelScope into local cache
#   - Every subsequent call: returns the cached path immediately (no network)
#   - Cache: ~/.cache/modelscope/hub/models/BAAI/bge-m3/
#
# WHY use snapshot_download() instead of a hardcoded path?
#   snapshot_download() always returns the correct absolute path regardless of OS.
#   It also handles the case where the cache is in a custom location.
#
# WHY pass the local path to BGEM3FlagModel instead of "BAAI/bge-m3"?
#   BGEM3FlagModel("BAAI/bge-m3") calls HuggingFace internally on every startup
#   to check for updates — which fails if HuggingFace is blocked or slow.
#   Passing a local path loads from disk directly, zero network calls.
from modelscope import snapshot_download
from FlagEmbedding import BGEM3FlagModel

# Loaded ONCE at server startup — no network calls after first download
_model_path = snapshot_download("BAAI/bge-m3")
_model = BGEM3FlagModel(_model_path, use_fp16=True)
```

**Option 2 — Ollama (removes FlagEmbedding + torch entirely):**

[Ollama](https://ollama.com) serves BGE-M3 as a local HTTP service. Fits naturally into a Docker Compose setup alongside Milvus.

_Local machine:_

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh    # macOS

# Ollama starts automatically as a background service after install.
# If you see "address already in use", it is already running — that is fine.
ollama pull bge-m3

# Verify
curl http://localhost:11434/api/embeddings \
  -d '{"model": "bge-m3", "prompt": "Hello world"}'
# → {"embedding": [0.023, -0.417, ...]}  (1024 floats)
```

_Docker Compose_ (Milvus + Ollama together):

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

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama # model persists across restarts

volumes:
  milvus_data:
  ollama_data:
```

```bash
docker compose up -d

# Pull BGE-M3 into the container (run once — ~2.3GB)
docker exec -it ollama ollama pull bge-m3
```

`embeddings.py` becomes a simple async HTTP call — no model loaded in the Python process:

```python
# src/rag/embeddings.py — Ollama variant
import httpx


class EmbeddingClient:
    DIMENSIONS = 1024
    # Requires Ollama running: `ollama serve` or via Docker Compose

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "bge-m3", "prompt": text},
            )
            return response.json()["embedding"]  # list of 1024 floats
```

> **Ollama caveat:** Dense vectors only, input silently truncated to 4,096 tokens. Sparse and ColBERT modes are not available. Sufficient for standard RAG — not recommended if you need hybrid retrieval or long document support (>4,096 tokens).

#### Which option should you choose?

|                                    | **Option 1 — ModelScope + FlagEmbedding**    | **Option 2 — Ollama**       |
| ---------------------------------- | -------------------------------------------- | --------------------------- |
| Already confirmed working          | Need to update `embeddings.py` to local path | Yes — `curl` test passed ✓  |
| Server startup time                | ~10s (loads 2GB into RAM)                    | Instant (HTTP call)         |
| RAM in FastAPI process             | ~1 GB                                        | ~0 (runs in Ollama process) |
| `torch` / `FlagEmbedding` required | Yes                                          | No                          |
| Docker Compose fit                 | N/A                                          | Natural (alongside Milvus)  |
| Dense RAG (standard use)           | Yes                                          | Yes                         |
| Sparse + ColBERT modes             | Yes                                          | No                          |
| Max input tokens                   | 8,192                                        | 4,096 (truncated silently)  |

**Recommendation: Ollama** if you are running standard dense RAG and already use Docker for Milvus. It removes ~1GB of Python dependencies, starts instantly, and fits your existing Docker workflow. Choose ModelScope if you later need hybrid retrieval (sparse + ColBERT) or documents longer than 4,096 tokens — both require a full Milvus server upgrade anyway.

#### Local experiment vs. production deployment

How you run BGE-M3 depends on the goal:

|                        | **FlagEmbedding**              | **Transformers + FastAPI** | **vLLM**                      |
| ---------------------- | ------------------------------ | -------------------------- | ----------------------------- |
| **Best for**           | Local experiment, learning     | Production API service     | High-concurrency / ARM server |
| **Setup effort**       | Minimal                        | Medium                     | Medium                        |
| **All 3 output modes** | Yes (dense + sparse + ColBERT) | Yes                        | Dense only                    |
| **Throughput**         | Standard PyTorch               | Good (ONNX optional)       | Highest                       |
| **Serves as HTTP API** | No                             | Yes (REST)                 | Yes (OpenAI-compatible)       |

**Local experiment → FlagEmbedding** (what this project uses)

The official library. Minimal code, all three retrieval modes work out of the box:

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output = model.encode(
    ["如何学习Python", "Python入门教程"],
    return_dense=True,
    return_sparse=True,       # keyword matching
    return_colbert_vecs=True, # token-level fine-grained matching
)
similarity = output['dense_vecs'][0] @ output['dense_vecs'][1].T
print(f"Semantic similarity: {similarity}")
```

**Production deployment → Transformers + FastAPI**

Load the model once at startup, expose it as a REST endpoint. Use the **local model path** (not the model ID string) so the server never calls HuggingFace:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch

app = FastAPI()

model_path = "/path/to/cached/bge-m3"  # local path, avoids network on startup
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
model.eval()

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    inputs = tokenizer(
        request.texts, padding=True, truncation=True,
        max_length=8192, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    vecs = torch.nn.functional.normalize(
        outputs.last_hidden_state.mean(dim=1), p=2, dim=1
    )
    return {"embeddings": vecs.cpu().tolist()}
```

```bash
uvicorn main:app --host 0.0.0.0 --port 33330 --workers 1
```

> **Production tips:** Convert to ONNX + INT8 quantization for CPU inference (several times faster, half the memory). Use `systemd` or `supervisor` to manage the process in production.

**Notes**

- **Hardware minimum:** 16 GB RAM. GPU recommended (4 GB+ VRAM); CPU works but is slower (~300–600 ms per call).
- **Long text:** BGE-M3 supports up to 8,192 tokens — a major advantage over most models. Always set `max_length=8192` when processing long documents. The default truncation (512) silently discards most of the text.
- **Ollama caveat:** Ollama's BGE-M3 support is limited — it outputs dense vectors only and silently truncates input to 4,096 tokens. This loses the sparse and ColBERT modes and half the context window. Not recommended for production RAG.

---

## 6. Switching from OpenAI to BGE-M3 — Migration Steps

```text
1. Replace src/rag/embeddings.py  →  use Option B code above

2. Update Milvus collection schema
   └── Change dim=1536 → dim=1024 everywhere

3. Reset Milvus Lite database (dimension mismatch = crash otherwise)
   └── Delete ./milvus_local.db — the server recreates it on next startup

4. Re-ingest all documents
   └── POST /v1/rag/ingest for each document
   └── The old vectors (1536-dim) are incompatible with the new model (1024-dim)

5. Remove OPENAI_API_KEY from .env (no longer needed)
```

```python
# Quick reset for Milvus Lite — run this once before restarting the server
import os
if os.path.exists("./milvus_local.db"):
    os.remove("./milvus_local.db")
    print("Done. Restart the server — Milvus Lite will recreate with the correct schema.")
```

---

## 7. Files

The only file that changes when switching embedding models:

```
app-ai/
└── src/
    └── rag/
        └── embeddings.py    ← swap the implementation here
```

Everything else — `retriever.py`, `vector_memory.py`, `routers/rag.py` — calls `EmbeddingClient().embed(text)` unchanged. The interface is identical regardless of which model is underneath.

Cross-references:

- [agent-rag.md — Section 4](agent-rag.md) — how embeddings fit into the full RAG pipeline
- [agent-rag.md — Section 5](agent-rag.md) — Milvus collection schema and vector dimensions
- [agent-memory.md](agent-memory.md) — memory system (uses the same EmbeddingClient)
