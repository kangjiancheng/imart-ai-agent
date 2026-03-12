from dataclasses import dataclass

from src.rag.embeddings import EmbeddingClient
from src.rag.milvus_utils import ensure_collection
from src.config.settings import settings


@dataclass
class Document:
    content: str
    source: str
    score: float


class RAGRetriever:
    TOP_K = settings.rag_top_k
    MIN_SCORE = settings.rag_min_score

    def __init__(self):
        self.embedder = EmbeddingClient()

    async def retrieve(self, query: str) -> list[Document]:
        """Search the knowledge_base collection for documents similar to the query."""
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )
            ensure_collection(client, settings.milvus_collection_knowledge)

            embedding = await self.embedder.embed(query)
            results = client.search(
                collection_name=settings.milvus_collection_knowledge,
                data=[embedding],
                limit=self.TOP_K,
                output_fields=["content", "source"],
            )

            docs = []
            for hit in results[0]:
                score = hit.get("distance", 0)
                if score >= self.MIN_SCORE:
                    docs.append(Document(
                        content=hit["entity"]["content"],
                        source=hit["entity"].get("source", "unknown"),
                        score=score,
                    ))
            return docs

        except Exception:
            return []  # Graceful degradation — agent continues without RAG context

    def format_for_prompt(self, docs: list[Document]) -> str:
        """Format retrieved documents into a string for the ToolMessage."""
        if not docs:
            return "No relevant documents found in the knowledge base."
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[Document {i} — source: {doc.source}]\n{doc.content}")
        return "\n\n---\n\n".join(parts)
