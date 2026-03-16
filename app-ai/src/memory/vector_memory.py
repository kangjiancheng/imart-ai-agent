# src/memory/vector_memory.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS VECTOR MEMORY?
#
# Vector Memory is the agent's LONG-TERM MEMORY across sessions.
# Unlike conversation `history` (which only covers the current session),
# vector memory persists facts about a user ACROSS many separate sessions.
#
# EXAMPLE FLOW:
#   Session 1: User says "I'm a senior engineer, I prefer Python."
#     → After the session, store_if_new() saves this fact to Milvus.
#   Session 2 (days later): User asks a new question.
#     → recall() retrieves "prefers Python" → Claude personalizes the answer.
#
# HOW IT DIFFERS FROM RAG (rag/retriever.py):
#   RAG (retriever.py)     = company KNOWLEDGE BASE — shared documents for all users
#   Vector Memory (here)   = per-USER MEMORY       — personal facts per user_id
#
#   Technical difference:
#     RAG collection:    "knowledge_base" — no user_id filter
#     Memory collection: "user_memory"    — filtered by user_id on every search
#
# WHERE IS IT USED?
#   agent_loop.py:
#     1. BEFORE the loop: recall() fetches relevant memories for system prompt
#     2. AFTER streaming: store_if_new() saves a summary of what was learned
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
# @dataclass decorator — auto-generates __init__, __repr__, __eq__.
# See agent/token_budget.py for a full explanation.

from src.rag.embeddings import EmbeddingClient
# EmbeddingClient wraps OpenAI's embedding API.
# Memory chunks are stored as vectors (same technique as RAG).

from src.rag.milvus_utils import ensure_collection
# ensure_collection(client, name) creates the Milvus collection with the
# standard schema if it doesn't exist yet. Shared across retriever, ingest,
# and vector memory to avoid duplicating the schema/index setup code.

from src.config.settings import settings
# settings = singleton Settings for milvus_host, milvus_port, collection names.


# ── MemoryChunk dataclass ──────────────────────────────────────────────────────
# Represents a single memory fact retrieved for a user.

@dataclass
class MemoryChunk:
    # content: the text of the remembered fact
    # e.g. "User prefers Python and works as a senior engineer."
    content: str

    # score: how semantically similar this memory is to the current query
    # Only memories with score >= MIN_SCORE (0.78) are returned.
    score: float


# ── VectorMemory ───────────────────────────────────────────────────────────────
# Main class for reading and writing per-user long-term memory.

class VectorMemory:
    # COLLECTION: the Milvus collection name for user memories.
    # "user_memory" — a separate collection from "knowledge_base" used by RAG.
    # Class-level constant: shared across all VectorMemory instances.
    COLLECTION = settings.milvus_collection_memory  # "user_memory"

    # MIN_SCORE: minimum similarity threshold for memory recall.
    # Set to 0.40 — personal memory queries are often indirect ("Who am I?", "What do I do?")
    # and score lower than exact-match queries, so a strict threshold causes misses.
    # WHY NOT lower than 0.40? Below this, unrelated memories start appearing as noise.
    MIN_SCORE = 0.40

    def __init__(self):
        # Create one EmbeddingClient per VectorMemory instance.
        # The embedder converts memory text and queries into vectors.
        self.embedder = EmbeddingClient()

    async def recall(
        self, user_id: str, query: str, top_k: int = 5
    ) -> list[str]:
        """
        Retrieve memory chunks relevant to this query for a specific user.
        Returns a list of plain strings (the memory content).
        Returns [] if Milvus is unavailable — loop continues without personalization.

        PARAMETERS:
          user_id: str  — scopes the search to this user's memories only
          query: str    — the current user message (used to find relevant memories)
          top_k: int = 5 — max results to return; default 5 if not specified

        PYTHON CONCEPT — default parameter value:
          `top_k: int = 5` means the caller can omit this argument.
          recall(user_id, query)         → top_k is 5
          recall(user_id, query, top_k=3) → top_k is 3
          In TypeScript: recall(userId: string, query: string, topK = 5)

        RETURN TYPE — list[str]:
          A simple list of text strings (not MemoryChunk objects).
          The strings are injected directly into the system prompt.
          MemoryChunk is defined above but used internally if needed.

        EFFECT ON THE AGENT:
          Returned strings are passed to build_system_prompt() in claude_client.py.
          Claude reads them at the start of every loop iteration as background context.
          This gives Claude "memory" of past sessions without re-training.
        """
        try:
            # PYTHON CONCEPT — deferred import:
            #   `from pymilvus import MilvusClient` inside try block.
            #   If pymilvus is missing or Milvus is offline → caught by except → return [].
            #   App startup never crashes due to Milvus being unavailable.
            from pymilvus import MilvusClient

            # Milvus Lite: uri is a local .db file path — no Docker needed.
            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )

            # Create the collection on first use if it doesn't exist yet.
            ensure_collection(client, self.COLLECTION)

            # Step 1: Embed the current query to find relevant memories.
            # await pauses until OpenAI returns the embedding vector.
            embedding = await self.embedder.embed(query)

            # Step 2: Search Milvus, filtered to this specific user.
            results = client.search(
                collection_name=self.COLLECTION,
                # "user_memory" — different collection from knowledge_base.

                data=[embedding],
                # The query vector. List of one because we search one query at a time.

                limit=top_k,
                # How many results to return at most (default 5).

                filter=f'user_id == "{user_id}"',
                # KEY DIFFERENCE FROM RAG: this filter scopes the search to one user.
                # Without this, user A's memories would appear in user B's searches.
                # f'user_id == "{user_id}"' = Milvus filter expression string.
                # Example: 'user_id == "user-abc-123"'
                # This is Milvus query language — similar to SQL WHERE clause.

                output_fields=["content"],
                # Only return the content field (we don't need tags or timestamps here).
            )

            # Step 3: Filter by MIN_SCORE and collect content strings.
            chunks = []
            for hit in results[0]:
                # PYTHON CONCEPT — dict.get() with default:
                #   hit.get("distance", 0) reads the "distance" key.
                #   If missing (Milvus version difference), defaults to 0.
                score = hit.get("distance", 0)
                content = hit["entity"]["content"]
                print(f"[recall] score={score:.4f} MIN={self.MIN_SCORE} content='{content[:60]}'")
                if score >= self.MIN_SCORE:
                    chunks.append(content)
                    # Append just the content string — the system prompt
                    # doesn't need the score or metadata, just the text.
            return chunks

        except Exception:
            # Any failure (Milvus down, connection timeout, etc.) returns []
            # instead of crashing. The agent loop continues without personalization.
            return []

    async def store_if_new(
        self, user_id: str, content: str, tags: list[str]
    ) -> None:
        """
        Store a memory chunk for this user.
        Skips silently if Milvus is unavailable — the session still completes.

        PARAMETERS:
          user_id: str       — which user this memory belongs to
          content: str       — the text to remember (e.g. a session summary)
          tags: list[str]    — metadata tags for future filtering (e.g. ["preference"])

        PYTHON CONCEPT — return type None:
          `-> None` means this function doesn't return anything useful.
          It performs a side effect (writing to Milvus) and returns nothing.
          In TypeScript: (): void => { ... }

        PYTHON CONCEPT — `pass` in except block:
          `except Exception: pass` means "catch any exception and do nothing."
          Memory write failure is deliberately ignored — it's non-fatal.
          The user's answer was already streamed; losing the memory write is acceptable.
          In TypeScript: catch (e) { /* intentionally empty */ }

        WHEN IS THIS CALLED?
          agent_loop.py calls store_if_new() AFTER the final answer is streamed.
          The agent loop already yielded all tokens to the user before this runs.
          A Milvus failure here doesn't affect the user's experience at all.
        """
        try:
            from pymilvus import MilvusClient
            import time
            # time = Python's built-in module for time-related functions.
            # time.time() returns the current Unix timestamp as a float (seconds).
            # int(time.time()) converts to integer (e.g. 1709042400).
            # Used to record when this memory was created.

            # Milvus Lite: uri is a local .db file path — no Docker needed.
            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )

            # Create the collection on first use if it doesn't exist yet.
            ensure_collection(client, self.COLLECTION)

            # Step 1: Embed the memory content for vector storage.
            # This embedding is what Milvus searches against later in recall().
            embedding = await self.embedder.embed(content)

            # Step 2: Insert the memory record into Milvus.
            client.insert(
                collection_name=self.COLLECTION,
                data=[{
                    # PYTHON CONCEPT — dict literal inside a list:
                    #   data=[{...}] = a list containing one dict (one record to insert).
                    #   Milvus accepts batch inserts, so it expects a list.
                    "user_id": user_id,
                    # Which user this memory belongs to.
                    # Used as the filter in recall() → `user_id == "{user_id}"`

                    "content": content,
                    # The actual text to remember.

                    "tags": tags,
                    # Metadata tags for potential future filtering.
                    # e.g. ["preference", "technical"] or ["order_history"]

                    "created_at": int(time.time()),
                    # Unix timestamp as integer.
                    # int() truncates the float: 1709042400.7 → 1709042400

                    "vector": embedding,
                    # The 1536-float embedding vector for this content.
                    # This is what Milvus searches when recall() is called.
                }],
            )
        except Exception:
            pass  # Memory write failure is non-fatal — silently ignored.
