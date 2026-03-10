# src/rag/embeddings.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS AN EMBEDDING?
#
# An embedding converts text into a list of numbers (a "vector").
# Example: "Hello world" → [0.023, -0.417, 0.891, ..., 0.142]  (1024 numbers)
#
# WHY? Computers can't directly compare meaning in two sentences.
# But they CAN measure the distance between two vectors.
# Sentences with similar meaning produce vectors that are CLOSE to each other.
#
#   "What is machine learning?" → [0.21, 0.87, -0.43, ...]
#   "Explain AI training"       → [0.19, 0.91, -0.41, ...]  ← similar direction!
#   "What's the weather today?" → [-0.72, 0.03, 0.65, ...]  ← very different
#
# This is the mathematical foundation of RAG (Retrieval-Augmented Generation):
#   1. At ingest time: embed every document chunk → store in Milvus
#   2. At query time: embed the user's question → find nearby vectors in Milvus
#   3. Nearby = semantically similar → retrieve those documents for Claude
#
# WHY BGE-M3 via ModelScope?
#   - ModelScope (Alibaba) hosts the same model weights as HuggingFace
#     but without network issues (.DS_Store 403, timeouts, blocked regions).
#   - snapshot_download() returns the local cache path on repeat calls —
#     no network access after the first download.
#   - The model runs entirely on your machine: free, private, multilingual.
#   - Anthropic has no embeddings API, so a separate model is always required.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
# asyncio = Python's standard library for async programming.
# We need asyncio.get_event_loop() to run a blocking function in a thread pool.

from functools import partial
# partial(fn, arg1, arg2) returns a new callable with those arguments pre-filled.
# run_in_executor() only accepts a zero-argument callable, so we use partial()
# to bake in the arguments ahead of time.

from modelscope import snapshot_download
# snapshot_download('BAAI/bge-m3'):
#   - First call: downloads ~2.3GB from ModelScope into local cache
#   - Every subsequent call: returns the cached path immediately (no network)
#   - Cache location: ~/.cache/modelscope/hub/models/BAAI/bge-m3/
#
# WHY ModelScope instead of passing "BAAI/bge-m3" directly to BGEM3FlagModel?
#   BGEM3FlagModel("BAAI/bge-m3") calls HuggingFace internally on every startup
#   to check for updates — which fails if HuggingFace is blocked or slow.
#   Passing a local path bypasses all network calls entirely.

from FlagEmbedding import BGEM3FlagModel
# BGEM3FlagModel = the official wrapper for BGE-M3 from BAAI.
# Accepts either a model ID string (triggers HuggingFace) or a local path (offline).


# ── Module-level singleton ────────────────────────────────────────────────────
# WHY a module-level variable (not inside __init__)?
#   Loading BGE-M3 takes ~10 seconds and uses ~1GB of RAM (with fp16).
#   A module-level variable is created ONCE when this module is first imported
#   (at server startup) and reused for every subsequent request.
#   In TypeScript: like a module-level const initialized once at import time.
#
# snapshot_download() is called here at module load time.
# On first run: downloads the model (~2.3GB). On every run after: instant cache hit.
_model_path = snapshot_download("BAAI/bge-m3")
# _model_path = "~/.cache/modelscope/hub/models/BAAI/bge-m3" (resolved absolute path)

_model = BGEM3FlagModel(_model_path, use_fp16=True)
# use_fp16=True: store weights as 16-bit floats instead of 32-bit.
#   → halves memory usage (~1GB instead of ~2GB), with minimal quality loss.
# Passing _model_path (local path) instead of "BAAI/bge-m3" (model ID):
#   → FlagEmbedding loads from disk directly, no HuggingFace network call.


# ── EmbeddingClient ───────────────────────────────────────────────────────────
# A class that wraps the BGE-M3 embedding call behind a clean async interface.
#
# WHY A CLASS instead of a plain function?
#   All callers (retriever.py, vector_memory.py, routers/rag.py) call
#   EmbeddingClient().embed(text) — they don't need to know which model is used.
#   The interface stays identical whether the model is BGE-M3, OpenAI, or Ollama.
#
# WHY run_in_executor?
#   FlagEmbedding's encode() is SYNCHRONOUS — it blocks the calling thread
#   until the model finishes computing. Calling it directly inside `async def`
#   would freeze FastAPI's entire event loop, preventing any other requests
#   from being processed while one embedding is being computed.
#   run_in_executor() offloads the blocking call to a separate thread pool,
#   so the event loop stays free to handle other requests concurrently.
#   In TypeScript: like wrapping a blocking call in a Worker thread.

class EmbeddingClient:
    DIMENSIONS = 1024
    # BGE-M3's dense vectors are 1024-dimensional.
    # This constant is read by the Milvus collection schema setup code
    # to know how wide each vector field must be.

    async def embed(self, text: str) -> list[float]:
        """Convert text to a 1024-dimensional dense vector using BGE-M3.

        RETURN TYPE — list[float]:
          A list of 1024 floating-point numbers representing the semantic
          meaning of the input text. In TypeScript: number[]

        PYTHON CONCEPT — async def / await:
          `async def embed(...)` declares this as a coroutine function.
          Calling embed("hello") doesn't run immediately — it returns a coroutine.
          `await embed("hello")` runs the coroutine and waits for the result.
          This allows FastAPI to handle other requests while BGE-M3 computes.
          In TypeScript: async embed(text: string): Promise<number[]>

        WHEN IS THIS CALLED?
          RAGRetriever.retrieve() calls it to embed the user's query.
          VectorMemory.recall() calls it to embed the memory search query.
          VectorMemory.store_if_new() calls it to embed the memory to store.
          rag.py's ingest endpoint calls it to embed each document chunk.
        """
        loop = asyncio.get_event_loop()
        # get_event_loop() returns the currently running asyncio event loop.
        # We need it to call run_in_executor(), which schedules work on a thread pool.

        fn = partial(_model.encode, [text], batch_size=1, max_length=512)
        # partial() pre-fills the arguments for _model.encode().
        # _model.encode() signature: encode(sentences, batch_size, max_length, ...)
        #   sentences=[text]  → list with one string (encode always takes a list)
        #   batch_size=1      → process one text at a time (we only have one)
        #   max_length=512    → truncate input at 512 tokens if longer
        #                       (BGE-M3 supports up to 8,192, but 512 is fast
        #                       and covers most query/chunk sizes in practice)

        result = await loop.run_in_executor(None, fn)
        # run_in_executor(None, fn):
        #   None → use the default thread pool executor (managed by Python)
        #   fn   → call fn() in a worker thread
        # `await` suspends this coroutine until fn() finishes in the thread.
        # During that suspension, the event loop is FREE to process other requests.

        # result is a dict with keys: 'dense_vecs', 'lexical_weights', 'colbert_vecs'
        # We use only 'dense_vecs' — the standard semantic search vector.
        # result['dense_vecs'] shape: (1, 1024) — one row per input text.
        #   [0]       → takes the first (and only) row → shape (1024,)
        #   .tolist() → converts numpy array to plain Python list[float]
        #               Milvus expects list[float], not numpy arrays.
        return result["dense_vecs"][0].tolist()
