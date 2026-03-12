import asyncio
from functools import partial

from modelscope import snapshot_download
from FlagEmbedding import BGEM3FlagModel

# Load BGE-M3 once at startup (~10s, ~1GB RAM with fp16).
# snapshot_download returns cached path on repeat calls — no network after first run.
_model_path = snapshot_download("BAAI/bge-m3")
_model = BGEM3FlagModel(_model_path, use_fp16=True)


class EmbeddingClient:
    DIMENSIONS = 1024  # BGE-M3 dense vector size

    async def embed(self, text: str) -> list[float]:
        """Convert text to a 1024-dimensional dense vector using BGE-M3.

        Runs the blocking encode() call in a thread pool to avoid freezing
        FastAPI's event loop.
        """
        loop = asyncio.get_event_loop()
        fn = partial(_model.encode, [text], batch_size=1, max_length=512)
        result = await loop.run_in_executor(None, fn)
        return result["dense_vecs"][0].tolist()
