# Stub — optional Cohere cross-encoder re-ranker
# Activate by installing: pip install cohere


def rerank(query: str, docs: list[str]) -> list[str]:
    """
    Re-rank documents by relevance to the query using a cross-encoder.
    Returns docs in descending relevance order.
    Falls back to original order if Cohere is unavailable.
    """
    try:
        import cohere
        co = cohere.Client()
        results = co.rerank(query=query, documents=docs, model="rerank-english-v3.0")
        return [docs[r.index] for r in results.results]
    except Exception:
        return docs
