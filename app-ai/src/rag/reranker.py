# src/rag/reranker.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS A RERANKER?
#
# RAG retrieval happens in TWO stages:
#
#   Stage 1 — Vector search (fast, approximate)
#     Query → embed → Milvus ANN search → Top-K candidates (e.g. 10 chunks)
#     Uses cosine similarity of compressed vectors.
#     Good at finding "same topic" but can miss precise relevance.
#
#   Stage 2 — Reranker (slower, precise)  ← THIS FILE
#     Takes the Top-K candidates → scores each one against the FULL query
#     → re-sorts by true relevance → returns only the best N to Claude.
#
# WHY DOES VECTOR SEARCH ALONE FAIL SOMETIMES?
#
#   Query: "What is the refund policy for digital products?"
#
#   Milvus returns (by cosine similarity alone):
#     #1  "Refunds take 3-5 business days"          score=0.81  ← relevant ✓
#     #2  "Digital products include e-books"         score=0.79  ← topic match, not policy
#     #3  "Return policy covers physical items only" score=0.77  ← WRONG answer
#
#   After reranking:
#     #1  "Refunds take 3-5 business days"          → 0.92  ← promoted, actually answers query
#     #3  "Return policy covers physical items only" → 0.61  ← demoted, contradicts query
#     #2  "Digital products include e-books"         → 0.44  ← demoted, not about policy
#
# WHY IS A CROSS-ENCODER MORE ACCURATE THAN COSINE SIMILARITY?
#
#   Bi-encoder (what embeddings.py does):
#     query  → encode separately → query_vector
#     doc    → encode separately → doc_vector
#     score  = cosine(query_vector, doc_vector)
#     FAST because vectors are pre-computed and stored in Milvus.
#     LESS ACCURATE because query and doc never "see" each other during encoding.
#
#   Cross-encoder (what rerankers do):
#     score  = model([query + doc] concatenated as one input)
#     SLOW because the model reads query + doc together every time.
#     MORE ACCURATE because attention layers see both texts simultaneously.
#
# STRATEGY IN THIS FILE:
#   Primary  → FlagReranker (BGE-M3 family, local, free, multilingual)
#   Fallback → Cohere API (optional, English-focused, best quality)
#   Default  → original order (if both are unavailable)
#
# WHY BGE-RERANKER AS PRIMARY?
#   - Same BAAI/BGE model family as your BGE-M3 embedder — consistent quality.
#   - Runs locally: free, private, no API key needed.
#   - Multilingual: handles Korean, Japanese, Chinese, English equally well.
#   - Already available via FlagEmbedding package (already in requirements.txt).
#   Model: BAAI/bge-reranker-v2-m3 (~1.1 GB, downloaded on first use via ModelScope)
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
# asyncio = Python's standard library for async/concurrent programming.
# We need asyncio.get_event_loop() to offload the blocking reranker call
# to a thread pool so FastAPI's event loop is not blocked.

from functools import partial
# partial(fn, arg1, arg2) = returns a new callable with arguments pre-filled.
# run_in_executor() only accepts a zero-argument callable, so we use partial()
# to bake in the query and docs arguments ahead of time.

# ── Module-level singleton for the local BGE reranker ─────────────────────────
# WHY eager-load at module import time (not lazy)?
#
#   PROBLEM WITH LAZY LOADING (loading on first request):
#     Every time uvicorn restarts (e.g. --reload on file save), module-level
#     variables reset to None. The NEXT request then triggers a re-load from disk,
#     which prints the "Downloading Model..." log line and adds latency to that
#     first request. This is confusing and slows down the first real user request.
#
#   SOLUTION — eager load at import time (same pattern as embeddings.py):
#     When Python imports this module (at server startup), it immediately runs
#     the code at module level — including snapshot_download() and FlagReranker().
#     The model is in RAM before the first request ever arrives.
#     The "Downloading Model..." log appears once during startup, not during requests.
#
#   SAME PATTERN AS embeddings.py:
#     _model_path = snapshot_download("BAAI/bge-m3")   ← runs at import time
#     _model = BGEM3FlagModel(_model_path, ...)         ← runs at import time
#
#   TRADEOFF:
#     Server startup is ~5 seconds slower (model load time).
#     But every request after that is instant — no per-request load cost.
#
# The leading underscore _ = Python convention for "private to this module".

def _load_bge_reranker():
    """
    Load the BGE reranker model from local ModelScope cache.
    Called once at module import time. Returns None if unavailable.

    PYTHON CONCEPT — module-level function called immediately:
      This function is defined AND called on the next line:
        _bge_reranker = _load_bge_reranker()
      It runs once when Python first imports this file (server startup).
      After that, _bge_reranker holds the loaded model for the process lifetime.
    """
    try:
        from modelscope import snapshot_download
        # snapshot_download: checks local cache first.
        # First ever call: downloads ~1.1 GB from ModelScope.
        # Every subsequent call: returns cached path instantly (no network).
        # Cache location: ~/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3/

        from FlagEmbedding import FlagReranker
        # FlagReranker = cross-encoder model from the BAAI/BGE family.
        # Unlike BGEM3FlagModel (bi-encoder), FlagReranker reads
        # [query, document] pairs together and outputs a single relevance score.

        model_path = snapshot_download("BAAI/bge-reranker-v2-m3")
        # "BAAI/bge-reranker-v2-m3":
        #   - v2 = second generation (better quality than v1)
        #   - m3 = multilingual (same family as BGE-M3 embedder)
        #   ~1.1 GB on disk; fp16 halves RAM to ~550 MB.

        return FlagReranker(model_path, use_fp16=True)
        # use_fp16=True = 16-bit weights → half the RAM, minimal quality loss.
        # Passing model_path (local dir) avoids any HuggingFace network calls.

    except Exception:
        # FlagEmbedding not installed, or ModelScope cache missing.
        # Return None — rerank() will fall back to Cohere or original order.
        return None


_bge_reranker = _load_bge_reranker()
# ↑ Called immediately at module import time.
# _bge_reranker is now either a FlagReranker instance or None.
# No lazy loading — model is ready before the first request arrives.


def _get_bge_reranker():
    """
    Returns the module-level FlagReranker singleton.
    Simple accessor — no loading logic here anymore.
    Kept as a function so tests can patch it with mock.patch().

    WHY KEEP THIS AS A FUNCTION (not just read _bge_reranker directly)?
      Tests use: patch("src.rag.reranker._get_bge_reranker", return_value=mock)
      If rerank() read _bge_reranker directly, tests would need to patch the
      module-level variable instead, which is slightly less clean.
      A function call is easier to intercept with unittest.mock.
    """
    return _bge_reranker


async def rerank(query: str, docs: list[str]) -> list[str]:
    """
    Re-rank a list of document strings by their relevance to the query.
    Returns the same documents in descending relevance order.

    PARAMETERS:
      query: str       — the user's original question
      docs:  list[str] — the document content strings to re-rank
                         (these come from Document.content in retriever.py)

    RETURN TYPE — list[str]:
      The same strings, reordered. Most relevant first.
      If all methods fail, returns the original order (safe fallback).

    PYTHON CONCEPT — async def:
      This function is a coroutine — the caller must `await` it.
      We need async because the BGE reranker is a BLOCKING call (CPU-bound)
      and we use run_in_executor() to run it in a thread pool without
      blocking FastAPI's event loop.
      In TypeScript: async rerank(query: string, docs: string[]): Promise<string[]>

    PRIORITY ORDER:
      1. Local BGE reranker (free, private, multilingual)
      2. Cohere API (optional, higher quality for English, requires COHERE_API_KEY)
      3. Original order (fallback — always safe)
    """
    # Nothing to rerank — return immediately.
    if not docs:
        return docs

    # ── Strategy 1: Local BGE reranker ────────────────────────────────────────
    reranker = _get_bge_reranker()
    if reranker is not None:
        try:
            # Build [query, doc] pairs for the cross-encoder.
            # FlagReranker.compute_scores() expects a list of [query, doc] pairs:
            #   [[query, doc1], [query, doc2], ...]
            # It returns a list of float scores, one per pair.
            pairs = [[query, doc] for doc in docs]
            # PYTHON CONCEPT — list comprehension:
            #   [expression for item in iterable]
            #   = creates a new list by applying `expression` to each `item`.
            # pairs = [[query, docs[0]], [query, docs[1]], ...]
            # In TypeScript: docs.map(doc => [query, doc])

            loop = asyncio.get_event_loop()
            # get_event_loop() = returns the currently running asyncio event loop.
            # We need it to call run_in_executor() below.

            fn = partial(reranker.compute_scores, pairs, normalize=True)
            # partial() pre-fills the arguments so run_in_executor can call fn()
            # with no arguments (as it requires a zero-argument callable).
            # normalize=True → converts raw logits to 0.0–1.0 scores using sigmoid.
            #   Without normalize: raw scores like [-3.2, 1.7, 0.4] (hard to interpret)
            #   With normalize:    scores like [0.04, 0.85, 0.60] (easy to interpret)

            scores = await loop.run_in_executor(None, fn)
            # run_in_executor(None, fn):
            #   None → use Python's default thread pool
            #   fn   → call fn() in a worker thread (doesn't block the event loop)
            # `await` suspends this coroutine until the thread finishes.
            # scores = [0.04, 0.85, 0.60, ...] — one float per doc

            # Sort docs by score descending (highest relevance first).
            ranked = sorted(
                zip(docs, scores),
                # zip(docs, scores) = pairs each doc with its score:
                #   [(doc1, 0.04), (doc2, 0.85), (doc3, 0.60), ...]
                # In TypeScript: docs.map((doc, i) => [doc, scores[i]])

                key=lambda pair: pair[1],
                # lambda pair: pair[1]  = extract the score (index 1) from each pair
                # sorted() uses this as the comparison key.
                # In TypeScript: .sort((a, b) => b[1] - a[1])

                reverse=True,
                # reverse=True → descending order (highest score first)
            )

            return [doc for doc, _ in ranked]
            # List comprehension: extract just the doc string from each (doc, score) pair.
            # The score was only needed for sorting — discard it after.
            # _ = Python convention for "I don't need this variable".
            # In TypeScript: ranked.map(([doc, _]) => doc)

        except Exception:
            # BGE reranker failed (OOM, model error, etc.) — fall through to Cohere.
            pass

    # ── Strategy 2: Cohere API (optional upgrade) ─────────────────────────────
    # Cohere's rerank models are state-of-the-art for English but require:
    #   1. `pip install cohere` (not in requirements.txt by default)
    #   2. COHERE_API_KEY set in .env
    # If neither is present, this block is silently skipped.
    try:
        import cohere
        # Deferred import: only attempted if BGE failed.
        # ImportError is caught by except if cohere is not installed.

        import os
        api_key = os.getenv("COHERE_API_KEY")
        # os.getenv() reads an environment variable. Returns None if not set.
        # We check explicitly so we can skip Cohere rather than getting a
        # misleading "invalid API key" error from the Cohere client.

        if api_key:
            co = cohere.ClientV2(api_key=api_key)
            # ClientV2 = Cohere's current SDK version (v2 API).
            # Passing api_key explicitly rather than relying on the env var
            # being read internally — more predictable behaviour.

            response = co.rerank(
                model="rerank-v3.5",
                # rerank-v3.5 = Cohere's latest multilingual reranker (2024).
                # Supports 100+ languages, including Korean.
                # Upgrade from the old "rerank-english-v3.0" in the original stub.

                query=query,
                documents=docs,
                # documents = the list of strings to re-rank.
                # Cohere accepts plain strings — no need to wrap in dicts.

                top_n=len(docs),
                # top_n = how many results to return.
                # We want all docs back (just reordered), not a truncated subset.
                # top_n=len(docs) = return every doc, reordered.
            )

            return [docs[r.index] for r in response.results]
            # response.results = list of RerankResponseResultsItem objects,
            # each with `.index` (original position) and `.relevance_score`.
            # They are already sorted by relevance (highest first).
            # docs[r.index] = look up the original doc string by its original position.
            # In TypeScript: response.results.map(r => docs[r.index])

    except Exception:
        # Cohere not installed, API key missing, or API error — fall through.
        pass

    # ── Strategy 3: Fallback — original Milvus order ──────────────────────────
    # If both BGE and Cohere failed, return docs in the original order.
    # Milvus already sorted them by cosine similarity (highest first),
    # so the original order is still a reasonable approximation.
    return docs
