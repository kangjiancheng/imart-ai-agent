# src/rag/retriever.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS RAG RETRIEVAL?
#
# RAG = Retrieval-Augmented Generation.
# Instead of relying only on Claude's training data, the agent can look up
# information from a private knowledge base (e.g. product documentation,
# internal policies) stored in Milvus (a vector database).
#
# HOW RAG WORKS IN THIS SYSTEM:
#   1. At ingest time (rag.py router):
#      Document → chunked into paragraphs → each chunk embedded as a vector
#      → stored in Milvus "knowledge_base" collection
#
#   2. At query time (this file):
#      User's question → embedded as a vector → search Milvus for nearby vectors
#      → retrieve the most semantically similar text chunks
#      → Claude reads those chunks + answers more accurately
#
# WHEN IS retriever.py CALLED?
#   In agent_loop.py, when Claude decides to call the "rag_retrieve" tool,
#   the loop calls RAGRetriever().retrieve(query) instead of the registered stub.
#   The results are formatted and sent back to Claude as a ToolMessage.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
# @dataclass = a decorator that auto-generates __init__, __repr__, __eq__
# for a class based on its field annotations. See agent/token_budget.py
# for a full explanation of dataclass.

from src.rag.embeddings import EmbeddingClient
# EmbeddingClient = wraps OpenAI's embedding API (see embeddings.py)
# Used here to convert the user's query text → a vector for Milvus search.

from src.rag.reranker import rerank
# rerank(query, docs) = re-orders a list of document strings by relevance.
# Called AFTER Milvus search to improve the final ranking before Claude reads them.
# Falls back silently to original Milvus order if no reranker is available.
# See reranker.py for a full explanation of WHY reranking improves RAG quality.

from src.rag.milvus_utils import ensure_collection
# ensure_collection(client, name) creates the Milvus collection with the
# standard schema if it doesn't exist yet. Shared across retriever, ingest,
# and vector memory to avoid duplicating the schema/index setup code.

from src.config.settings import settings
# settings = the singleton Settings instance for config values like TOP_K, MIN_SCORE.


# ── Document dataclass ─────────────────────────────────────────────────────────
# Represents a single retrieved document chunk from the knowledge base.

@dataclass
class Document:
    # content: the text of the retrieved chunk (a paragraph of source document)
    content: str

    # source: where this chunk came from (filename, URL, etc.)
    # e.g. "product-manual.pdf", "api-reference.html"
    source: str

    # score: cosine similarity score from Milvus (0.0 → 1.0)
    # Higher = more semantically similar to the query.
    # We only keep chunks with score >= MIN_SCORE (0.72).
    score: float


# ── RAGRetriever ───────────────────────────────────────────────────────────────
# The main class for querying the knowledge base.
#
# PYTHON CONCEPT — class-level constants (vs instance variables):
#   Variables defined at class level (not inside __init__) are shared by all
#   instances. We read settings values here because settings is already loaded.
#   In TypeScript: class RAGRetriever { static readonly TOP_K = 5 }

class RAGRetriever:
    TOP_K = settings.rag_top_k          # 5 — how many results to return from Milvus
    MIN_SCORE = settings.rag_min_score  # 0.72 — discard results below this similarity

    def __init__(self):
        # Create one EmbeddingClient per RAGRetriever instance.
        # EmbeddingClient holds an AsyncOpenAI connection pool — reuse it.
        self.embedder = EmbeddingClient()

    async def retrieve(self, query: str) -> list[Document]:
        """
        Search the knowledge_base Milvus collection for documents
        semantically similar to the query.
        Returns only results above MIN_SCORE.

        RETURN TYPE:
          list[Document] = a list of Document dataclass instances.
          Returns [] (empty list) if Milvus is unavailable — graceful degradation.

        PYTHON CONCEPT — async def / await:
          This is a coroutine. The caller must `await` it:
            docs = await retriever.retrieve("user's query")
          While waiting for Milvus, other requests can run concurrently.
          In TypeScript: async retrieve(query: string): Promise<Document[]>

        EFFECT ON THE AGENT:
          Called from agent_loop.py when Claude uses the rag_retrieve tool.
          Returns document chunks that become the ToolMessage content.
          Claude reads them before writing its final answer.
        """
        try:
            # PYTHON CONCEPT — deferred import (local import inside function):
            #   `from pymilvus import MilvusClient` is inside the function body.
            #   If pymilvus is not installed, the import error is caught by `except`.
            #   The agent continues without RAG rather than crashing at startup.
            from pymilvus import MilvusClient

            # Milvus Lite: uri is a local file path (e.g. "./milvus_local.db").
            # pymilvus creates the file automatically — no Docker or server needed.
            # To use a full Milvus server instead, set MILVUS_URI=http://host:port in .env.
            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
                # token="user:password" for Milvus server auth (e.g. "root:Milvus").
                # None = no auth (used for Milvus Lite local file).
            )

            # Ensure the collection exists before searching.
            ensure_collection(client, settings.milvus_collection_knowledge)

            # Step 1: Convert the user's query text to a vector.
            # await = pause here until OpenAI returns the embedding (non-blocking).
            embedding = await self.embedder.embed(query)

            # Step 2: Search Milvus for vectors close to the query vector.
            results = client.search(
                collection_name=settings.milvus_collection_knowledge,
                # The Milvus collection name — "knowledge_base" by default.

                data=[embedding],
                # data = list of query vectors. We search for one query at a time.
                # Milvus accepts batch searches, so it expects a list.
                # [embedding] = a list containing our single query vector.

                limit=self.TOP_K,
                # Maximum number of results to return (5).
                # Milvus returns the TOP_K most similar vectors.

                output_fields=["content", "source"],
                # Which fields to include in each result besides the vector.
                # "content" = the text chunk, "source" = the original document name.
            )
            # results is a list of lists: results[0] = list of hit dicts for query 0.
            # Since we searched for one query, we read results[0].

            # Step 3: Filter results and build Document objects.
            docs = []
            # `docs = []` initializes an empty list to collect results.
            # In TypeScript: const docs: Document[] = []

            for hit in results[0]:
                # PYTHON CONCEPT — for loop:
                #   `for hit in results[0]` iterates over each search result.
                #   `hit` is a dict with keys: "distance", "id", "entity".
                #   In TypeScript: for (const hit of results[0])

                score = hit.get("distance", 0)
                # hit.get("distance", 0) = read the "distance" key; default 0 if missing.
                # In Milvus with cosine similarity, "distance" IS the similarity score.
                # Score range: 0.0 (no similarity) → 1.0 (identical).

                if score >= self.MIN_SCORE:
                    # Only include chunks above the similarity threshold (0.72).
                    # Chunks below this are likely not relevant to the query.
                    docs.append(Document(
                        # Document(...) = create a Document dataclass instance.
                        # @dataclass auto-generated __init__ accepts keyword args.
                        content=hit["entity"]["content"],
                        # hit["entity"] = the stored fields for this result.
                        # hit["entity"]["content"] = the text of the chunk.

                        source=hit["entity"].get("source", "unknown"),
                        # .get("source", "unknown") = safe read with fallback.
                        # Some chunks might not have a source field → "unknown".

                        score=score,
                    ))
            # ── Step 4: Rerank for precision ──────────────────────────────────
            # Milvus sorted docs by cosine similarity (approximate).
            # Reranking re-scores them using a cross-encoder that reads
            # query + document TOGETHER — more accurate than vector distance alone.
            #
            # rerank() returns the same content strings in a new order.
            # We rebuild Document objects to preserve source/score metadata.
            #
            # WHY rerank AFTER the MIN_SCORE filter (not before)?
            #   We already dropped low-quality results. Reranking a smaller,
            #   pre-filtered list is faster and produces cleaner results.
            if docs:
                contents = [doc.content for doc in docs]
                # Extract just the text strings — rerank() works on plain strings.
                # In TypeScript: docs.map(d => d.content)

                reranked_contents = await rerank(query, contents)
                # await = pause here until the BGE reranker (or Cohere) finishes.
                # reranked_contents = same strings, now in relevance order.

                # Build a lookup dict so we can reassemble Document objects
                # with original source/score after reordering.
                # {content_string: Document} — lets us find metadata by content.
                doc_map = {doc.content: doc for doc in docs}
                # PYTHON CONCEPT — dict comprehension:
                #   {key_expr: val_expr for item in iterable}
                # In TypeScript: Object.fromEntries(docs.map(d => [d.content, d]))

                docs = [doc_map[content] for content in reranked_contents if content in doc_map]
                # Reassemble docs in the new reranked order.
                # `if content in doc_map` guards against any edge case where
                # the reranker returns a string not in our original set.

            return docs

        except Exception:
            # PYTHON CONCEPT — bare except clause:
            #   `except Exception:` catches ANY exception without binding it to a variable.
            #   We don't need the error details here — Milvus being down is not fatal.
            #   Milvus being unavailable is not fatal — the agent continues
            #   answering from Claude's training data only.
            return []
            # Returning [] means the agent gets no RAG context.
            # Claude will answer from its training knowledge only.
            # No crash, no 500 error — graceful degradation.

    def format_for_prompt(self, docs: list[Document]) -> str:
        """Format retrieved documents into a string for the ToolMessage.

        WHEN IS THIS CALLED?
          After retrieve() returns docs, agent_loop.py calls format_for_prompt()
          to convert the list of Document objects into a single readable string.
          That string becomes the content of a ToolMessage sent to Claude.

        PYTHON CONCEPT — non-async def:
          This is a REGULAR (synchronous) function, not async.
          No I/O happens here — just string formatting — so no await needed.
          In TypeScript: just a regular method returning string.

        RETURN TYPE — str:
          A multi-paragraph string Claude can read in the conversation context.
        """
        if not docs:
            # `not docs` = True when docs is an empty list [].
            # Empty list is "falsy" in Python — just like [] in JavaScript.
            return "No relevant documents found in the knowledge base."

        parts = []
        # parts = a list of formatted string chunks, one per document.
        # We'll join them all at the end.

        for i, doc in enumerate(docs, 1):
            # PYTHON CONCEPT — enumerate(iterable, start):
            #   enumerate() yields (index, value) pairs during iteration.
            #   The `start=1` argument makes the index start at 1 (not 0).
            #   `for i, doc in enumerate(docs, 1)` unpacks each pair:
            #     First iteration:  i=1, doc=docs[0]
            #     Second iteration: i=2, doc=docs[1]
            #     etc.
            #   In TypeScript: docs.forEach((doc, idx) => { const i = idx + 1 })

            parts.append(f"[Document {i} — source: {doc.source}]\n{doc.content}")
            # f"[Document {i} — source: {doc.source}]\n{doc.content}"
            # = a formatted string showing the document number, source, and content.
            # \n = newline character (line break).
            # Example output:
            #   [Document 1 — source: api-reference.pdf]
            #   The API accepts POST requests with a JSON body...

        return "\n\n---\n\n".join(parts)
        # "\n\n---\n\n".join(parts)
        # = join all document strings with a horizontal rule separator between them.
        # "\n\n---\n\n" = blank line + "---" + blank line (Markdown hr).
        # This makes it easy for Claude to visually separate one document from another.
        # In TypeScript: parts.join("\n\n---\n\n")
