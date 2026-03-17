from dataclasses import dataclass

from src.rag.embeddings import EmbeddingClient
from src.rag.milvus_utils import ensure_collection
from src.config.settings import settings


@dataclass
class MemoryChunk:
    """A single memory fact retrieved for a user."""
    content: str
    score: float


class VectorMemory:
    """
    Per-user long-term memory stored in the Milvus "user_memory" collection.

    Differs from RAGRetriever (knowledge_base):
      - Scoped per user_id — each search is filtered to one user.
      - Lower MIN_SCORE (0.40) because personal queries are often indirect.
      - Gracefully degrades to [] on Milvus failure — agent continues without personalization.

    agent_loop.py calls:
      recall()        — before the loop, to inject memory into the system prompt.
      store_if_new()  — after streaming, to persist facts and session summaries.
    """

    COLLECTION = settings.milvus_collection_memory
    MIN_SCORE  = 0.40

    def __init__(self):
        self.embedder = EmbeddingClient()

    async def recall(self, user_id: str, query: str, top_k: int = 5) -> list[str]:
        """
        Search user_memory for facts relevant to the current query.

        Returns a list of plain strings injected into build_system_prompt().
        Returns [] if Milvus is unavailable — agent continues without personalization.
        """
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )
            ensure_collection(client, self.COLLECTION)

            embedding = await self.embedder.embed(query)

            results = client.search(
                collection_name=self.COLLECTION,
                data=[embedding],
                limit=top_k,
                filter=f'user_id == "{user_id}"',
                output_fields=["content"],
            )

            chunks = []
            for hit in results[0]:
                score   = hit.get("distance", 0)
                content = hit["entity"]["content"]
                if score >= self.MIN_SCORE:
                    chunks.append(content)
            return chunks

        except Exception:
            return []

    async def store_if_new(self, user_id: str, content: str, tags: list[str]) -> None:
        """
        Embed and insert a memory chunk into user_memory.

        Called after the agent response is fully streamed — Milvus write latency
        never affects the user. Silently ignores failures (non-fatal).
        """
        try:
            from pymilvus import MilvusClient
            import time

            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )
            ensure_collection(client, self.COLLECTION)

            embedding = await self.embedder.embed(content)

            client.insert(
                collection_name=self.COLLECTION,
                data=[{
                    "user_id":    user_id,
                    "content":    content,
                    "tags":       tags,
                    "created_at": int(time.time()),
                    "vector":     embedding,
                }],
            )
        except Exception:
            pass
