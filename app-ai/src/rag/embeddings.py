# src/rag/embeddings.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS AN EMBEDDING?
#
# An embedding converts text into a list of numbers (a "vector").
# Example: "Hello world" → [0.023, -0.417, 0.891, ..., 0.142]  (1536 numbers)
#
# WHY? Computers can't directly compare meaning in two sentences.
# But they CAN measure the distance between two vectors.
# Sentences with similar meaning produce vectors that are CLOSE to each other.
#
#   "What is machine learning?" → [0.21, 0.87, -0.43, ...]
#   "Explain AI training"       → [0.19, 0.91, -0.41, ...]  ← similar direction!
#   "What's the weather today?" → [-0.72, 0.03, 0.65, ...]  ← very different
#
# This is the mathematical foundation of RAG (Retrieval-Augmented Generation):
#   1. At ingest time: embed every document chunk → store in Milvus
#   2. At query time: embed the user's question → find nearby vectors in Milvus
#   3. Nearby = semantically similar → retrieve those documents for Claude
#
# WHY OPENAI'S EMBEDDING MODEL (not Anthropic's)?
#   As of the project's design, Anthropic does not offer a standalone embedding
#   API. OpenAI's text-embedding-3-small is fast, cheap, and high-quality.
#   The embedding model is completely separate from the LLM — Claude is still
#   the language model generating responses; OpenAI is only used for embeddings.
# ─────────────────────────────────────────────────────────────────────────────

from openai import AsyncOpenAI
# AsyncOpenAI = the async version of the OpenAI Python client.
# "Async" means the .embeddings.create() call doesn't BLOCK the event loop
# while waiting for the API response — other requests can run concurrently.
# In TypeScript: like using fetch() instead of XMLHttpRequest.sync().

from src.config.settings import settings
# settings = the singleton Settings instance.
# Used to read settings.openai_api_key and settings.embedding_model.


# ── EmbeddingClient ───────────────────────────────────────────────────────────
# A class that wraps the OpenAI embedding API call.
#
# PYTHON CONCEPT — class with __init__:
#   __init__(self) is the constructor — runs when you write EmbeddingClient().
#   `self` is Python's explicit reference to the current instance.
#   In TypeScript: class EmbeddingClient { constructor() { this.client = ... } }
#
# WHY A CLASS instead of a plain function?
#   The AsyncOpenAI client maintains connection pools and configuration.
#   Creating it once in __init__ and reusing it in embed() is more efficient
#   than creating a new client on every function call.

class EmbeddingClient:
    # CLASS-LEVEL CONSTANT:
    # MODEL is defined at class level (not inside __init__), so it's shared
    # by all instances. Reading settings at class definition time is fine
    # because settings is already loaded by the time this module is imported.
    # In TypeScript: static readonly MODEL = settings.embedding_model
    MODEL = settings.embedding_model  # "text-embedding-3-small"

    def __init__(self):
        # Create one AsyncOpenAI client per EmbeddingClient instance.
        # self.client is an INSTANCE variable — each EmbeddingClient object
        # gets its own client attribute.
        # In TypeScript: this.client = new AsyncOpenAI({ apiKey: ... })
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed(self, text: str) -> list[float]:
        """Convert text to a vector (list of floats).

        RETURN TYPE — list[float]:
          A list of 1536 floating-point numbers (for text-embedding-3-small).
          Each number represents one dimension of the semantic meaning vector.
          In TypeScript: number[]

        PYTHON CONCEPT — async def / await:
          `async def embed(...)` declares this as a coroutine function.
          Calling embed("hello") doesn't run immediately — it returns a coroutine.
          `await embed("hello")` runs the coroutine and waits for the result.
          This allows other async tasks to run while waiting for OpenAI's API.
          In TypeScript: async embed(text: string): Promise<number[]>

        WHEN IS THIS CALLED?
          RAGRetriever.retrieve() calls it to embed the user's query.
          VectorMemory.recall() calls it to embed the memory search query.
          VectorMemory.store_if_new() calls it to embed the memory to store.
          rag.py's ingest endpoint calls it to embed each document chunk.
        """
        response = await self.client.embeddings.create(
            # self.client.embeddings.create() = the OpenAI embedding API call.
            # `await` means: pause this coroutine until OpenAI responds, then continue.
            # The event loop can process other requests during this pause.
            model=self.MODEL,
            # self.MODEL = "text-embedding-3-small"
            # This model produces 1536-dimensional vectors.
            # text-embedding-3-large produces 3072-dim (higher quality, slower, costlier).
            input=text,
            # The text to convert to a vector.
        )
        # response.data[0].embedding = the 1536-float vector for the input text.
        #   response.data = list of EmbeddingObject (one per input, we sent one)
        #   response.data[0] = the first (and only) embedding result
        #   .embedding = the actual list[float] vector
        # In TypeScript: response.data[0].embedding
        return response.data[0].embedding
