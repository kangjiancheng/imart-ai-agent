# tests/unit/test_reranker.py
#
# Unit tests for src/rag/reranker.py
#
# WHAT WE TEST:
#   - rerank() returns the same documents (no content lost or added)
#   - rerank() changes the order so the most relevant doc is first
#   - rerank() handles edge cases: empty list, single doc, duplicates
#   - rerank() falls back gracefully when the BGE model is unavailable
#
# HOW TO RUN (from app-ai/ directory with venv active):
#   pytest tests/unit/test_reranker.py -v
#
# PYTHON CONCEPT — pytest:
#   pytest discovers test files named test_*.py and runs functions named test_*().
#   No class needed — plain functions work fine.
#   `assert` statements are the checks. If an assert fails, pytest shows what
#   the actual value was vs. what was expected.
#
# PYTHON CONCEPT — pytest.mark.asyncio:
#   rerank() is an `async def` function — it must be awaited.
#   pytest can't await async functions by default.
#   `@pytest.mark.asyncio` tells pytest to run the test inside an event loop.
#   Requires: pip install pytest-asyncio  (add to requirements.txt if missing)

import pytest
from unittest.mock import patch, MagicMock
# unittest.mock = Python's built-in mocking library.
# patch()     = temporarily replaces a real object with a fake one during a test.
# MagicMock() = a fake object that accepts any attribute access or method call.
# WHY MOCK?
#   The real BGE reranker downloads ~1.1 GB and takes seconds to load.
#   Unit tests must run in milliseconds without network or GPU access.
#   We mock the model so tests verify our LOGIC, not the ML model's correctness.

from src.rag.reranker import rerank


# ── helpers ───────────────────────────────────────────────────────────────────

def make_docs() -> list[str]:
    """
    A realistic set of 4 document strings simulating Milvus search results
    for the query "What is the refund policy for digital products?".

    Ordered as Milvus would return them (by cosine similarity, not true relevance):
      [0] about refunds (relevant but generic)
      [1] about digital products (topic match, but not about policy)
      [2] about physical return policy (misleading — physical, not digital)
      [3] about digital refund specifically (most relevant — should become #1)
    """
    return [
        "Refunds are processed within 3-5 business days after approval.",
        "Digital products include software licenses, e-books, and online courses.",
        "Our return policy covers physical items only. Items must be unused.",
        "Customers may request a refund within 30 days of purchase for digital items.",
    ]


# ── Test 1: fallback returns original order when BGE is unavailable ───────────

@pytest.mark.asyncio
async def test_rerank_fallback_returns_original_order():
    """
    When the BGE reranker model is unavailable (e.g. not downloaded yet,
    or FlagEmbedding not installed), rerank() must return docs in the
    original order — not crash, not return an empty list.

    HOW THE MOCK WORKS:
      `patch("src.rag.reranker._get_bge_reranker", return_value=None)`
      Temporarily replaces _get_bge_reranker() with a function that returns None.
      This simulates the model being unavailable.
      After the `with` block exits, the real function is restored automatically.
    """
    docs = make_docs()

    with patch("src.rag.reranker._get_bge_reranker", return_value=None):
        # Also patch cohere so it doesn't accidentally succeed
        with patch.dict("sys.modules", {"cohere": None}):
            result = await rerank("What is the refund policy for digital products?", docs)

    # Must return all 4 docs — no content dropped
    assert len(result) == len(docs)

    # Must be the original order (fallback path)
    assert result == docs


# ── Test 2: rerank returns all docs (no content lost) ─────────────────────────

@pytest.mark.asyncio
async def test_rerank_returns_all_docs():
    """
    After reranking, every document from the input must appear in the output.
    The reranker must never silently drop a document.

    We mock compute_scores to return fixed scores so the test is deterministic.
    Fixed scores: [0.3, 0.2, 0.15, 0.9]
    Expected order after sort descending: [doc[3], doc[0], doc[1], doc[2]]
    """
    docs = make_docs()
    # Scores assigned to each doc in input order:
    # doc[0]=0.3, doc[1]=0.2, doc[2]=0.15, doc[3]=0.9
    mock_scores = [0.3, 0.2, 0.15, 0.9]

    mock_model = MagicMock()
    mock_model.compute_scores.return_value = mock_scores
    # MagicMock() creates a fake object.
    # .compute_scores.return_value = mock_scores means:
    #   whenever mock_model.compute_scores(...) is called, return mock_scores.
    # Our reranker calls: reranker.compute_scores(pairs, normalize=True)
    # The mock ignores the arguments and always returns mock_scores.

    with patch("src.rag.reranker._get_bge_reranker", return_value=mock_model):
        result = await rerank("What is the refund policy for digital products?", docs)

    # All 4 docs must be present — set comparison ignores order
    assert set(result) == set(docs)

    # Exact count — no duplicates added, no docs dropped
    assert len(result) == 4


# ── Test 3: rerank puts highest-scored doc first ──────────────────────────────

@pytest.mark.asyncio
async def test_rerank_most_relevant_doc_is_first():
    """
    The document with the highest reranker score must end up at position 0.

    We mock scores so doc[3] (the most relevant) gets score 0.95.
    After reranking, result[0] must be docs[3].
    """
    docs = make_docs()
    # doc[3] = "Customers may request a refund within 30 days for digital items."
    # This is the most relevant answer — give it the highest score.
    mock_scores = [0.40, 0.25, 0.10, 0.95]
    #              doc0   doc1   doc2   doc3 ← should become result[0]

    mock_model = MagicMock()
    mock_model.compute_scores.return_value = mock_scores

    with patch("src.rag.reranker._get_bge_reranker", return_value=mock_model):
        result = await rerank("What is the refund policy for digital products?", docs)

    assert result[0] == docs[3], (
        f"Expected most relevant doc first.\n"
        f"Got: {result[0]}\n"
        f"Expected: {docs[3]}"
    )


# ── Test 4: rerank puts lowest-scored doc last ────────────────────────────────

@pytest.mark.asyncio
async def test_rerank_least_relevant_doc_is_last():
    """
    The misleading doc (physical return policy) should end up last
    because the reranker correctly scores it lowest for this query.
    """
    docs = make_docs()
    # doc[2] = "Our return policy covers physical items only."
    # This is misleading for a digital products query — give it the lowest score.
    mock_scores = [0.50, 0.45, 0.05, 0.90]
    #              doc0   doc1  doc2   doc3
    #                           ↑ lowest → should be result[-1]

    mock_model = MagicMock()
    mock_model.compute_scores.return_value = mock_scores

    with patch("src.rag.reranker._get_bge_reranker", return_value=mock_model):
        result = await rerank("What is the refund policy for digital products?", docs)

    assert result[-1] == docs[2], (
        f"Expected least relevant doc last.\n"
        f"Got: {result[-1]}\n"
        f"Expected: {docs[2]}"
    )


# ── Test 5: empty input returns empty output ──────────────────────────────────

@pytest.mark.asyncio
async def test_rerank_empty_list_returns_empty():
    """
    Edge case: if retrieve() returns [] (Milvus found nothing),
    rerank() must return [] without error.
    """
    result = await rerank("any query", [])
    assert result == []


# ── Test 6: single doc returns unchanged ─────────────────────────────────────

@pytest.mark.asyncio
async def test_rerank_single_doc_returns_unchanged():
    """
    Edge case: one document — nothing to reorder.
    The result must be a list containing that one document.
    """
    docs = ["Only one document in the knowledge base."]
    mock_scores = [0.88]

    mock_model = MagicMock()
    mock_model.compute_scores.return_value = mock_scores

    with patch("src.rag.reranker._get_bge_reranker", return_value=mock_model):
        result = await rerank("any query", docs)

    assert result == docs


# ── Test 7: verify compute_scores is called with correct pairs ────────────────

@pytest.mark.asyncio
async def test_rerank_passes_correct_pairs_to_model():
    """
    The BGE reranker expects pairs: [[query, doc1], [query, doc2], ...]
    This test verifies our code builds those pairs correctly.

    WHY TEST THE CALL ARGUMENTS?
      If we pass the wrong format (e.g. just docs without the query),
      the model silently returns garbage scores. Catching this in a unit test
      is far easier than debugging wrong RAG answers in production.
    """
    query = "What is the refund policy for digital products?"
    docs = make_docs()
    mock_scores = [0.5, 0.4, 0.3, 0.2]

    mock_model = MagicMock()
    mock_model.compute_scores.return_value = mock_scores

    with patch("src.rag.reranker._get_bge_reranker", return_value=mock_model):
        await rerank(query, docs)

    # Verify compute_scores was called exactly once
    mock_model.compute_scores.assert_called_once()

    # Extract what arguments it was called with
    call_args = mock_model.compute_scores.call_args
    # call_args.args[0] = the first positional argument = the pairs list
    pairs_passed = call_args.args[0]

    # Every pair must be [query, doc]
    assert len(pairs_passed) == len(docs)
    for i, pair in enumerate(pairs_passed):
        assert pair[0] == query,  f"pair[{i}][0] should be the query"
        assert pair[1] == docs[i], f"pair[{i}][1] should be docs[{i}]"
