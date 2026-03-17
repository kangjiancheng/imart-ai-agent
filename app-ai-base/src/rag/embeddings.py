import asyncio
from functools import partial

from modelscope import snapshot_download
from FlagEmbedding import BGEM3FlagModel

# Module-level singleton — loaded once at server startup (~10 s, ~1 GB RAM with fp16).
# snapshot_download returns the local ModelScope cache path on repeat calls
# (no network after the first download).
_model_path = snapshot_download("BAAI/bge-m3")
_model      = BGEM3FlagModel(_model_path, use_fp16=True)


class EmbeddingClient:
    """
    Async wrapper around the BGE-M3 local embedding model.

    BGE-M3 produces 1024-dimensional dense vectors, free, multilingual,
    runs entirely on this machine. encode() is synchronous (CPU-bound),
    so it is offloaded to a thread pool via run_in_executor() to keep
    FastAPI's event loop free during embedding computation.
    """

    DIMENSIONS = 1024

    async def embed(self, text: str) -> list[float]:
        """
        Convert text to a 1024-dimensional dense vector.

        Called by RAGRetriever.retrieve(), VectorMemory.recall(),
        VectorMemory.store_if_new(), and the RAG ingest endpoint.
        """
        loop = asyncio.get_event_loop()
        fn   = partial(_model.encode, [text], batch_size=1, max_length=512)
        result = await loop.run_in_executor(None, fn)
        return result["dense_vecs"][0].tolist()
