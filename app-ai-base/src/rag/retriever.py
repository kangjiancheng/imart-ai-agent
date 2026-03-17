from dataclasses import dataclass

from src.rag.embeddings import EmbeddingClient
from src.rag.reranker import rerank
from src.rag.milvus_utils import ensure_collection
from src.config.settings import settings


@dataclass
class Document:
    """A single retrieved document chunk from the knowledge_base collection."""
    content: str
    source:  str
    score:   float


class RAGRetriever:
    """
    Queries the Milvus knowledge_base collection for documents semantically
    similar to a user query (RAG — Retrieval-Augmented Generation).

    Two-stage retrieval:
      1. Vector search (bi-encoder cosine similarity) — fast, approximate.
      2. Reranking (cross-encoder) — slower but more precise relevance scoring.

    Degrades gracefully to [] when Milvus is unavailable — Claude answers
    from training data only with no crash or error.
    """

    TOP_K     = settings.rag_top_k
    MIN_SCORE = settings.rag_min_score

    def __init__(self):
        self.embedder = EmbeddingClient()

    async def retrieve(self, query: str) -> list[Document]:
        """
        Embed the query, search Milvus, filter by MIN_SCORE, then rerank.

        Returns list[Document] sorted by relevance (most relevant first),
        or [] on failure.
        """
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

            if docs:
                contents         = [doc.content for doc in docs]
                reranked_contents = await rerank(query, contents)
                doc_map = {doc.content: doc for doc in docs}
                docs    = [doc_map[c] for c in reranked_contents if c in doc_map]

            return docs

        except Exception:
            return []

    def format_for_prompt(self, docs: list[Document]) -> str:
        """
        Format retrieved documents into a single string for the ToolMessage.

        Each document is prefixed with "[Document N — source: filename]" and
        separated by "---" so Claude can visually distinguish documents.
        """
        if not docs:
            return "No relevant documents found in the knowledge base."

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[Document {i} — source: {doc.source}]\n{doc.content}")
        return "\n\n---\n\n".join(parts)
