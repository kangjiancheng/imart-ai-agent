import asyncio
from functools import partial


def _load_bge_reranker():
    """
    Load the BGE reranker model from ModelScope cache at module import time.

    Returns a FlagReranker instance on success, or None if unavailable.
    Eager-loading (vs lazy) ensures the model is ready before the first request
    and avoids per-request startup latency.
    """
    try:
        from modelscope import snapshot_download
        from FlagEmbedding import FlagReranker

        model_path = snapshot_download("BAAI/bge-reranker-v2-m3")
        return FlagReranker(model_path, use_fp16=True)
    except Exception:
        return None


_bge_reranker = _load_bge_reranker()


def _get_bge_reranker():
    """Returns the module-level FlagReranker singleton. Kept as a function for test patching."""
    return _bge_reranker


async def rerank(query: str, docs: list[str]) -> list[str]:
    """
    Re-rank document strings by relevance to the query using a cross-encoder.

    Priority:
      1. Local BGE reranker (BAAI/bge-reranker-v2-m3) — free, multilingual, offline.
      2. Cohere API (optional) — requires `pip install cohere` and COHERE_API_KEY.
      3. Original Milvus order (fallback) — already sorted by cosine similarity.

    compute_scores() is synchronous (CPU-bound), offloaded to a thread pool via
    run_in_executor() to avoid blocking FastAPI's event loop.
    """
    if not docs:
        return docs

    # ── Strategy 1: Local BGE reranker ────────────────────────────────────────
    reranker = _get_bge_reranker()
    if reranker is not None:
        try:
            pairs = [[query, doc] for doc in docs]
            loop  = asyncio.get_event_loop()
            fn    = partial(reranker.compute_scores, pairs, normalize=True)
            scores = await loop.run_in_executor(None, fn)

            ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
            return [doc for doc, _ in ranked]
        except Exception:
            pass

    # ── Strategy 2: Cohere API ────────────────────────────────────────────────
    try:
        import cohere
        import os

        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            co       = cohere.ClientV2(api_key=api_key)
            response = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs,
                top_n=len(docs),
            )
            return [docs[r.index] for r in response.results]
    except Exception:
        pass

    # ── Strategy 3: Original order ────────────────────────────────────────────
    return docs
