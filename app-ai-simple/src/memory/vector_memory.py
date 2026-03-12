from dataclasses import dataclass

from src.rag.embeddings import EmbeddingClient
from src.rag.milvus_utils import ensure_collection
from src.config.settings import settings


@dataclass
class MemoryChunk:
    content: str
    score: float


class VectorMemory:
    COLLECTION = settings.milvus_collection_memory
    MIN_SCORE = 0.78  # Higher than RAG — wrong memories mislead Claude more than missing ones

    def __init__(self):
        self.embedder = EmbeddingClient()

    async def recall(self, user_id: str, query: str, top_k: int = 5) -> list[str]:
        """Retrieve memory chunks relevant to this query for a specific user."""
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
                if hit.get("distance", 0) >= self.MIN_SCORE:
                    chunks.append(hit["entity"]["content"])
            return chunks

        except Exception:
            return []  # Graceful degradation — agent continues without personalization

    async def store_if_new(self, user_id: str, content: str, tags: list[str]) -> None:
        """Store a memory chunk for this user. Silently skips on failure."""
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
                    "user_id": user_id,
                    "content": content,
                    "tags": tags,
                    "created_at": int(time.time()),
                    "vector": embedding,
                }],
            )
        except Exception:
            pass  # Memory write failure is non-fatal
