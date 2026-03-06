# Agent RAG System

> **Scope:** Deep-dive guide for the Retrieval-Augmented Generation (RAG) layer inside `app-ai`.
> For the full agent architecture, see [app-ai-architecture.md](app-ai-architecture.md).
> For the memory system (uses the same Milvus instance, different collection), see [agent-memory.md](agent-memory.md).

---

## 1. What Is RAG?

An LLM's training data has a **knowledge cutoff**. It cannot know about:

- Your company's internal documents
- Events after its training date
- Private data (product specs, support tickets, policies)
- Real-time information (prices, news, live metrics)

**RAG solves this** by retrieving relevant external content at query time and injecting it into the LLM prompt — giving the model grounded, up-to-date facts to reason over.

```text
Without RAG:                     With RAG:

User: "What is our refund        User: "What is our refund
       policy?"                         policy?"
LLM: "I don't have access               │
      to your policy documents"         ▼
                                 Retrieve: refund_policy_v3.pdf
                                        │
                                        ▼
                                 LLM: "According to our policy,
                                       refunds are processed within
                                       5 business days..."
```

---

## 2. RAG vs Memory vs LLM Knowledge

These three sources all feed the LLM but serve different purposes.

| Source | What it contains | Written by | Scope | Example |
|---|---|---|---|---|
| **LLM training** | General world knowledge | Anthropic (training) | Global, static | "What is TCP/IP?" |
| **RAG (knowledge_base)** | Domain documents, product content | Ingestion pipeline | Shared across all users | "What does our refund policy say?" |
| **Memory (user_memory)** | User preferences, past tasks | Agent post-response | Per user | "This user prefers bullet points" |

All three can contribute to a single LLM response. The Planner decides which sources to activate for a given question.

---

## 3. The RAG Pipeline — Step by Step

```text
User question: "What are the return policy exceptions?"
        │
        ▼
① Query preprocessing
  └── Clean, normalize, extract key terms
  └── Optionally expand: "return policy" → "refund, return, exchange policy"

        │
        ▼
② Embedding
  └── Convert question text → dense vector (e.g. 1536-dim float array)
  └── Same model used at ingest time — must match

        │
        ▼
③ Milvus similarity search (ANN)
  └── Approximate nearest neighbour across knowledge_base vectors
  └── Returns top-K candidates with cosine distance scores

        │
        ▼
④ Score filtering
  └── Discard any result below MIN_SCORE (0.72)
  └── Prevents hallucination from low-confidence matches

        │
        ▼
⑤ Re-ranking (optional, production)
  └── Cross-encoder re-ranks shortlisted results
  └── More expensive but more accurate than vector similarity alone

        │
        ▼
⑥ Context assembly
  └── Format retrieved chunks into LLM-readable text
  └── Include source metadata (document name, page, date)

        │
        ▼
⑦ Inject into LLM prompt
  └── Placed at position 4 in the prompt (after memory, before short-term memory)
  └── LLM answers using retrieved facts, not training data alone
```

---

## 4. Embedding

### Why embedding is the foundation

RAG depends entirely on embedding quality. A bad embedding model produces vectors that cluster unrelated content together, causing irrelevant results to score highly. The search is only as good as the embedding.

### Claude does not provide embeddings

Anthropic does not offer an embedding endpoint. Use a dedicated embedding model:

| Model | Provider | Dimensions | Best for |
|---|---|---|---|
| `text-embedding-3-small` | OpenAI | 1536 | General purpose, good cost/quality |
| `text-embedding-3-large` | OpenAI | 3072 | Higher accuracy, 2× cost |
| `embed-english-v3.0` | Cohere | 1024 | English-only, strong re-ranking support |
| `embed-multilingual-v3.0` | Cohere | 1024 | Multilingual content |

**Critical rule:** The embedding model used at **ingest time** (when documents are stored) and at **query time** (when the user asks a question) **must be identical**. Mixing models produces nonsense similarity scores.

### Embedding client

```python
# app/rag/embeddings.py
#
# An "embedding" converts text into a list of numbers (a vector) that captures
# semantic meaning. Similar sentences end up with similar vectors (close in space).
# This is what makes vector search possible — instead of matching keywords, Milvus
# finds documents whose MEANING is close to the query's meaning.

from openai import AsyncOpenAI           # OpenAI's async Python client (pip install openai)
from app.config.settings import settings  # centralized config (reads from .env)

class EmbeddingClient:
    MODEL = "text-embedding-3-small"   # 1536-dimensional vectors; good cost/quality balance

    def __init__(self):
        # AsyncOpenAI = non-blocking client; uses `await` so the agent loop isn't paused
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed(self, text: str) -> list[float]:
        # Normalize whitespace — newlines can confuse some embedding models
        text = text.replace("\n", " ").strip()
        response = await self.client.embeddings.create(
            model=self.MODEL,
            input=text,
        )
        # The API returns a list of embedding objects; we always send one text so index [0]
        return response.data[0].embedding   # a list of 1536 floats, e.g. [0.023, -0.147, ...]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding for ingest pipeline — more efficient than one-by-one."""
        # Clean all texts in a list comprehension (compact for-loop that builds a new list)
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        # One API call for all texts — much cheaper than N separate calls
        response = await self.client.embeddings.create(
            model=self.MODEL,
            input=cleaned,    # OpenAI accepts a list of strings in one request
        )
        # Return a list of vectors, one per input text, preserving order
        return [item.embedding for item in response.data]
```

---

## 5. Milvus Collection Design

### knowledge_base collection

Stores all ingested document chunks shared across users.

```python
# app/rag/schema.py
#
# This defines the shape of the Milvus collection — like a database table schema.
# Each FieldSchema is one column. The `vector` field is special: Milvus indexes it
# so it can find similar vectors in milliseconds across millions of documents.

from pymilvus import CollectionSchema, FieldSchema, DataType   # official Milvus Python client

KNOWLEDGE_BASE_SCHEMA = CollectionSchema(fields=[
    # Primary key — must be unique per document chunk.
    # We use a deterministic string ID (hash of source + chunk index) so re-ingestion is idempotent.
    FieldSchema(name="id",          dtype=DataType.VARCHAR,      max_length=64, is_primary=True),

    # THE core field: a 1536-dimensional float vector (matches text-embedding-3-small output).
    # Milvus builds an HNSW index on this field to enable fast approximate nearest-neighbour search.
    FieldSchema(name="vector",      dtype=DataType.FLOAT_VECTOR, dim=1536),

    # The actual text chunk — what gets injected into the LLM prompt
    FieldSchema(name="content",     dtype=DataType.VARCHAR,      max_length=4096),

    # Source metadata — shown to the LLM so it can cite documents ("According to refund_policy.pdf...")
    FieldSchema(name="source",      dtype=DataType.VARCHAR,      max_length=256),   # filename or URL
    FieldSchema(name="page",        dtype=DataType.INT32),         # page number in the original document
    FieldSchema(name="chunk_index", dtype=DataType.INT32),         # which chunk within the page

    FieldSchema(name="updated_at",  dtype=DataType.VARCHAR,      max_length=32),    # ISO timestamp string
    # ARRAY type: stores multiple string tags per chunk (e.g. ["policy", "refund", "v3"])
    FieldSchema(name="tags",        dtype=DataType.ARRAY,        element_type=DataType.VARCHAR, max_capacity=16),
])

# Index configuration for the `vector` field.
# HNSW (Hierarchical Navigable Small World) = best speed/accuracy tradeoff for most use cases.
# M=16: number of neighbours each node connects to (higher = more accurate, more memory)
# efConstruction=256: search depth during index build (higher = better quality index, slower build)
INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "COSINE",    # cosine similarity: measures angle between vectors (1.0 = identical)
    "params": {"M": 16, "efConstruction": 256},
}
```

### user_memory collection (for reference)

Separate collection, per-user filtered. See [agent-memory.md](agent-memory.md) for the full schema.

| Collection | Owner | Filter | GDPR scope |
|---|---|---|---|
| `knowledge_base` | Shared | None (all users) | Document-level deletion |
| `user_memory` | Per user | `user_id == '...'` | Full per-user deletion |

---

## 6. Retriever Implementation

```python
# app/rag/retriever.py
#
# RAGRetriever is the query side of the pipeline.
# Given a text question, it: embeds it → searches Milvus → filters low-score results → returns docs.
# The agent loop calls this when the Planner decides retrieval is needed.

from dataclasses import dataclass   # @dataclass auto-generates __init__ and __repr__
from pymilvus import MilvusClient
from app.rag.embeddings import EmbeddingClient
from app.config.settings import settings

# `@dataclass` turns this class into a simple data container.
# Without it, you'd have to write: def __init__(self, content, source, page, score): self.content = content ...
@dataclass
class Document:
    content:    str
    source:     str
    page:       int
    score:      float   # cosine similarity score (0.0–1.0, higher = more relevant to the query)

class RAGRetriever:
    COLLECTION = "knowledge_base"  # which Milvus collection to search (separate from user_memory)
    TOP_K      = 5                 # retrieve at most 5 candidates before score filtering
    MIN_SCORE  = 0.72              # discard results below this — prevents hallucination from weak matches

    def __init__(self):
        # MilvusClient connects to the vector database at startup
        self.milvus   = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        self.embedder = EmbeddingClient()

    # `tags` parameter is optional (None by default) — use it to narrow search to a document category
    async def retrieve(self, query: str, tags: list[str] | None = None) -> list[Document]:
        # Step 1: convert the query text to a vector (same model used at ingest time!)
        embedding = await self.embedder.embed(query)

        # Step 2: build an optional Milvus filter expression (like a SQL WHERE clause)
        filter_expr = f"tags in {tags}" if tags else ""   # empty string = no filter

        # Step 3: vector similarity search — Milvus finds the TOP_K closest vectors
        results = self.milvus.search(
            collection_name=self.COLLECTION,
            data=[embedding],          # list of query vectors (one here, but API accepts multiple)
            limit=self.TOP_K,          # return at most this many results
            filter=filter_expr,        # optional metadata filter
            output_fields=["content", "source", "page", "chunk_index"],  # fields to return
        )

        # Step 4: build Document objects, filtering out low-confidence matches
        # `results[0]` = results for the first (and only) query vector
        documents = [
            Document(
                content=r["entity"]["content"],
                source=r["entity"]["source"],
                page=r["entity"]["page"],
                score=r["distance"],   # "distance" in Milvus is actually the similarity score for COSINE
            )
            for r in results[0]
            if r["distance"] >= self.MIN_SCORE   # discard weak matches
        ]

        return documents    # already ordered by score descending (best match first)

    def format_for_prompt(self, documents: list[Document]) -> str:
        """Format retrieved documents into a prompt-injectable string."""
        if not documents:
            return "No relevant documents found."

        parts = []
        # `enumerate(documents, 1)` produces (1, doc1), (2, doc2), ... (starts at 1, not 0)
        for i, doc in enumerate(documents, 1):
            parts.append(
                f"[Source {i}: {doc.source}, page {doc.page}, score {doc.score:.2f}]\n"
                # :.2f formats the float to 2 decimal places: 0.8934 → "0.89"
                f"{doc.content}"
            )
        # "\n\n---\n\n".join(parts) inserts a separator between each document block
        return "\n\n---\n\n".join(parts)
```

---

## 7. Document Ingestion Pipeline

Documents must be processed and stored in Milvus before any retrieval can happen. This is a separate pipeline from the agent loop — it runs offline or on a schedule, never during a user request.

```text
Raw document (PDF, DOCX, TXT, HTML)
        │
        ▼
① Document loader
  └── Extract raw text from file format

        │
        ▼
② Text chunker
  └── Split into overlapping chunks (e.g. 512 tokens, 50-token overlap)
  └── Overlap prevents context loss at chunk boundaries

        │
        ▼
③ Batch embedding
  └── Embed all chunks in batches (OpenAI allows up to 2048 per call)

        │
        ▼
④ Milvus upsert
  └── Insert vectors + metadata into knowledge_base
  └── Use deterministic ID (hash of source + chunk_index) for idempotent re-ingestion

        │
        ▼
⑤ Confirm index
  └── Verify HNSW index is built before serving queries
```

### Chunking strategy

Chunking is one of the most impactful RAG decisions. Too large = noisy context. Too small = loses sentence meaning.

| Strategy | Chunk size | Overlap | Use when |
|---|---|---|---|
| Fixed token | 256–512 tokens | 50 tokens | General purpose, most common |
| Sentence | 3–5 sentences | 1 sentence | Prose-heavy documents (policies, articles) |
| Paragraph | One paragraph | None | Well-structured documents |
| Semantic | Variable | None | Highest quality, requires NLP model |

```python
# app/rag/chunker.py
#
# Before storing a document in Milvus, we split it into small chunks.
# Reason: embedding a 50-page PDF as one vector loses detail. Small chunks
# (256–512 tokens) produce more precise embeddings and more targeted retrieval.
# LangChain's RecursiveCharacterTextSplitter handles the splitting logic.

from langchain_text_splitters import RecursiveCharacterTextSplitter
# pip install langchain-text-splitters
# "Recursive" = tries to split on paragraph breaks first, then newlines, then sentences, then spaces.
# This preserves semantic boundaries rather than cutting mid-sentence.

class DocumentChunker:
    def chunk(self, text: str, source: str) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,            # target size in characters (not tokens — rough approximation)
            chunk_overlap=50,          # each chunk shares 50 chars with the next → no context lost at boundaries
            separators=["\n\n", "\n", ". ", " "],
            # ^ try splitting on: paragraph break → newline → sentence end → space
            # This ensures cuts happen at natural language boundaries when possible
        )
        chunks = splitter.split_text(text)  # returns a list of strings

        # Build a dict for each chunk — this becomes one Milvus record
        return [
            {
                "id":          f"{source}::chunk_{i}",  # deterministic ID → safe to re-ingest
                "content":     chunk,
                "source":      source,
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)  # enumerate gives (index, value) pairs: (0, "text...")
        ]
```

### Ingest endpoint

```python
# app/routers/rag.py
#
# This router handles document ingestion — uploading files so the agent can reference them.
# The key pattern: return a job_id immediately (202 Accepted), then process in the background.
# This prevents HTTP timeouts on large documents (PDF ingestion can take 10–30 seconds).

from fastapi import APIRouter, UploadFile, BackgroundTasks
# UploadFile = FastAPI's type for multipart file uploads
# BackgroundTasks = FastAPI's built-in task queue for post-response work

router = APIRouter(prefix="/v1")

@router.post("/rag/ingest")
async def ingest_document(
    file: UploadFile,                  # the uploaded file (PDF, DOCX, TXT, etc.)
    background_tasks: BackgroundTasks, # FastAPI injects this automatically
    tags: list[str] = [],              # optional category tags for filtering later
):
    """
    Called by app-service (NestJS) after file upload.
    Runs asynchronously — returns job_id immediately, processes in background.
    """
    job_id = generate_job_id()   # e.g. "job_a3f9b2c1"
    # `add_task` schedules `run_ingest_pipeline` to run AFTER this function returns the response.
    # The HTTP client gets the job_id instantly without waiting for the pipeline.
    background_tasks.add_task(run_ingest_pipeline, file, tags, job_id)
    return {"job_id": job_id, "status": "queued", "filename": file.filename}

# Polling endpoint — NestJS checks this to know when ingestion is done
@router.get("/rag/ingest/{job_id}")
async def ingest_status(job_id: str):
    # `{job_id}` in the path is a path parameter — FastAPI extracts it automatically
    status = await get_job_status(job_id)
    return status
```

---

## 8. When the Planner Activates RAG

RAG is not always needed. The Planner decides per-iteration whether to activate retrieval.

```text
Activate RAG when:
✅ Question requires factual grounding from documents
   "What does the SLA say about uptime guarantees?"
✅ Question references internal company content
   "What are the steps for employee onboarding?"
✅ LLM answer could be incorrect without domain context
   "What is our pricing for the enterprise tier?"

Skip RAG when:
❌ Question is answered by LLM general knowledge
   "What is REST API?"
❌ Answer already exists in short-term memory (prior iteration retrieved it)
❌ Token budget is exhausted — no room for retrieved content
❌ User is having a casual conversational exchange
   "Thanks, that was helpful!"
```

```python
# app/agent/planner.py (RAG decision)
#
# The Planner checks whether retrieval is needed BEFORE calling Claude.
# This fast keyword check avoids an unnecessary LLM call when the decision is obvious.
# For ambiguous cases, the LLM's Thought step makes the final call.

class Planner:
    # Keywords that strongly suggest the user is asking about internal documents.
    # Using a list of lowercase strings for fast substring matching.
    RAG_TRIGGER_KEYWORDS = [
        "policy", "document", "manual", "spec", "procedure",
        "according to", "what does", "where is", "how does our",
    ]

    def should_retrieve(self, message: str, memory: ShortMemory) -> bool:
        # If RAG already ran this loop, don't run it again — use what we already retrieved.
        # `memory.rag_documents` is a list; empty list is falsy in Python.
        if memory.rag_documents:
            return False

        # `.lower()` normalizes case so "Policy" matches "policy"
        lower = message.lower()

        # `any(...)` returns True if AT LEAST ONE keyword is found in the message.
        # This is the fast path — no LLM call needed.
        if any(kw in lower for kw in self.RAG_TRIGGER_KEYWORDS):
            return True

        # No keyword matched — defer to the LLM's Thought step for the final decision.
        # The LLM may still choose RAG_RETRIEVE even without a keyword match.
        return False
```

---

## 9. Score Threshold Tuning

`MIN_SCORE` is the most important RAG parameter. Setting it wrong causes either hallucination or no retrieval at all.

| Score range | Meaning | Risk |
|---|---|---|
| 0.90–1.00 | Near-exact match | Very safe, but misses paraphrased content |
| 0.80–0.89 | Strong semantic match | Good precision, recommended for strict domains |
| 0.72–0.79 | Moderate match | Balanced — the default in this codebase |
| 0.60–0.71 | Weak match | High hallucination risk — LLM may misuse context |
| < 0.60 | Noise | Never inject — worse than no retrieval |

**Tuning process:**

1. Collect 50–100 real user questions with known correct answers
2. Run retrieval at various thresholds (0.65, 0.70, 0.72, 0.75, 0.80)
3. Measure precision (were retrieved docs relevant?) and recall (were relevant docs retrieved?)
4. Choose the threshold that balances precision and recall for your document domain

---

## 10. Advanced: Hybrid Search

Pure vector search struggles with exact keyword matches (product codes, names, IDs). Hybrid search combines vector similarity with keyword (BM25) scoring.

```text
User: "Find the ERR_4291 error code documentation"
        │
        ├── Vector search: "error code documentation" → semantic results
        └── BM25 keyword search: "ERR_4291" → exact match results
                │
                ▼
        Reciprocal Rank Fusion (RRF)
        Merge and re-rank both result sets
                │
                ▼
        Final top-K passed to LLM
```

Use hybrid search when your documents contain:
- Product codes, SKUs, error codes, IDs
- Proper nouns (person names, place names, brand names)
- Structured data inline in prose

---

## 11. Advanced: Re-ranking

A cross-encoder re-ranker re-scores the top-K vector results with a more expensive but more accurate model. It reads the query and each document together (not as separate vectors), so it understands relevance in context.

```text
Vector search returns 10 candidates (fast, approximate)
        │
        ▼
Cross-encoder scores each (query, document) pair
        │
        ▼
Re-ranked top-5 passed to LLM (accurate, but adds ~200ms)
```

```python
# app/rag/reranker.py
#
# A cross-encoder re-ranker improves precision over pure vector search.
# Vector search finds "similar direction" in embedding space — fast but approximate.
# A cross-encoder reads the query AND each document together, understanding relevance in context.
# Trade-off: more accurate but adds ~150–300ms latency. Use in production when precision matters.

import cohere   # pip install cohere

class Reranker:
    MODEL = "rerank-english-v3.0"   # Cohere's cross-encoder model for English documents

    def __init__(self):
        # `cohere.Client` is synchronous; for async use `cohere.AsyncClient`
        self.client = cohere.Client(api_key=settings.cohere_api_key)

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        results = self.client.rerank(
            model=self.MODEL,
            query=query,
            documents=[d.content for d in documents],   # send just the text, not Document objects
            # [d.content for d in documents] = list comprehension: extract .content from each Document
            top_n=top_n,   # return only the top N after re-ranking (further reduces noise)
        )
        # `results.results` is a list of RerankResult objects, each with an `.index` into the original list.
        # We map index back to the original Document object to preserve metadata (source, page, score).
        return [documents[r.index] for r in results.results]
```

Add a re-ranker when:
- Retrieval precision is low (users report irrelevant answers)
- Documents contain many similar-sounding passages
- Latency budget allows for the extra 150–300ms

---

## 12. RAG Failure Modes and Handling

| Failure | Behaviour | Impact |
|---|---|---|
| Milvus unreachable | Return empty document list | LLM answers from training knowledge only — no crash |
| Zero results above MIN_SCORE | Return empty list | LLM answers without retrieval context |
| All results below MIN_SCORE | Filtered out | Same as above |
| Embedding API (OpenAI) timeout | Raise `EmbeddingError`, skip RAG | LLM proceeds without retrieval |
| Document chunk too large for prompt | Truncate to fit token budget | Partial context — log truncation event |
| Stale document (not yet re-indexed) | Returns old content | Add `updated_at` to prompt so LLM can note the date |

RAG retrieval is **always optional from the agent loop's perspective**. A retrieval failure returns an empty list; the agent continues reasoning without it. Retrieval failure never crashes the loop.

---

## 13. Production Checklist

```text
Embedding
  ✅ Same model at ingest and query time
  ✅ Batch embedding at ingest (not one-by-one)
  ✅ Normalize vectors before storing (cosine similarity requires it)

Chunking
  ✅ Overlap between chunks (50–100 tokens)
  ✅ Preserve sentence boundaries (do not cut mid-sentence)
  ✅ Store chunk_index and page for source citation

Milvus
  ✅ HNSW index built before serving
  ✅ knowledge_base and user_memory in separate collections
  ✅ Deterministic IDs for idempotent re-ingestion

Retrieval
  ✅ MIN_SCORE tuned for your document domain
  ✅ Graceful empty-result handling (no crash, no hallucination)
  ✅ Source metadata included in LLM prompt for citations

Ingest pipeline
  ✅ Runs as background task (never blocks the agent loop)
  ✅ Job status endpoint for app-service to poll
  ✅ Re-ingest is idempotent (upsert by deterministic ID)

Observability
  ✅ Log query, top-K scores, and count of results above threshold
  ✅ Track retrieval latency per request
  ✅ Alert when average MIN_SCORE drops (document quality degrading)
```

---

## 14. Files

```
app-ai/
└── app/
    ├── rag/
    │   ├── retriever.py     RAGRetriever — Milvus search, score filtering, prompt formatting
    │   ├── embeddings.py    EmbeddingClient — OpenAI text-embedding-3-small
    │   ├── chunker.py       DocumentChunker — text splitting with overlap
    │   └── reranker.py      Reranker — Cohere cross-encoder (optional, production)
    └── routers/
        └── rag.py           POST /v1/rag/ingest, GET /v1/rag/ingest/{job_id}
```

Cross-references:
- [app-ai-architecture.md — Component 6](app-ai-architecture.md) — summary and placement in the full agent
- [app-ai-architecture.md — Section 7](app-ai-architecture.md) — LLM prompt assembly order (RAG at position 4)
- [app-ai-architecture.md — Section 5](app-ai-architecture.md) — Token Budget (RAG skipped when budget exhausted)
- [agent-memory.md](agent-memory.md) — Memory system (uses same Milvus, different collection and purpose)
