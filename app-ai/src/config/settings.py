# src/config/settings.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is the single source of truth for ALL configuration values.
# It reads values from environment variables (and .env file) and validates them.
#
# WHY NOT JUST USE os.getenv() EVERYWHERE?
#
# In plain Python you could write:  os.getenv("ANTHROPIC_API_KEY")
# The problem: if the key is missing, you get None — and the error only appears
# much later when you actually try to use the key (at runtime, possibly in production).
#
# pydantic-settings solves this:
#   - Reads .env automatically at startup
#   - Validates that required keys exist (fails FAST with a clear error if missing)
#   - Converts types automatically (e.g. "19530" string → int 19530)
#   - Provides defaults for optional values
#
# This is the Python equivalent of TypeScript's zod + dotenv combination.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings
# BaseSettings = the base class from pydantic-settings that adds env-reading magic.
# Any class that inherits from BaseSettings gets this behavior automatically.


class Settings(BaseSettings):
    """
    All configuration for the app-ai service.

    PYTHON CONCEPT — class with typed fields:
    Each field below is:  name: type = default_value
    - If a field has no default, it is REQUIRED — the app won't start without it.
    - If a field has a default, it's OPTIONAL — the default is used if not in .env.

    pydantic-settings maps: field name → environment variable name (UPPERCASE).
      anthropic_api_key → reads ANTHROPIC_API_KEY from environment
      claude_model      → reads CLAUDE_MODEL from environment (or uses default)
    """

    # ── Claude ────────────────────────────────────────────────────────────────
    anthropic_api_key: str
    # No default = REQUIRED. If ANTHROPIC_API_KEY is not in .env, app crashes at startup
    # with a clear "field required" validation error. Better than crashing on first API call.

    claude_model: str = "claude-sonnet-4-6"
    # Default value — used unless you override CLAUDE_MODEL in .env.
    # claude-sonnet-4-6 = fast and smart, good balance of speed and quality.

    claude_max_tokens: int = 4096
    # Max tokens Claude can generate in its RESPONSE (not the input).
    # 4096 ≈ ~3,000 words — more than enough for most agent responses.

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Using BAAI/bge-m3 (local, free, multilingual) — no API key needed.
    # The model runs entirely on this machine; no external service is called.
    # Model is loaded once at server startup from the HuggingFace cache
    # (~10s startup, ~1GB RAM with fp16). See src/rag/embeddings.py.

    # ── Milvus vector database ────────────────────────────────────────────────
    milvus_uri: str = "./milvus_local.db"
    # milvus_uri: str # No default = REQUIRED. App won't start if MILVUS_URI is missing from .env
    # The milvus_uri from .env variable MILVUS_URI overrides this default.
    # milvus_uri: str = "./milvus_local.db": use Milvus Lite automatically" which lets the app work out of the box without any .env configuration.
    # Milvus Lite — an embedded, file-based Milvus that needs NO Docker or server.
    # The URI is just a local file path; pymilvus creates the file automatically.
    #
    # To switch to a full Milvus server (e.g. Docker), set in .env:
    #   MILVUS_URI=http://localhost:19530

    milvus_token: str = ""
    # Authentication token for Milvus server in "user:password" format.
    # Default Milvus credentials: root:Milvus
    # Set MILVUS_TOKEN=root:Milvus in .env when using Docker Milvus.
    # Leave empty ("") for Milvus Lite (no auth needed).

    milvus_collection_knowledge: str = "knowledge_base"
    # Milvus collection name for RAG documents (company docs, policies, etc.).
    # "Collection" in Milvus ≈ "table" in SQL or "collection" in MongoDB.

    milvus_collection_memory: str = "user_memory"
    # Separate collection for per-user long-term memory. Kept separate from knowledge_base
    # so GDPR deletion (forget one user's data) doesn't touch shared documents.

    # ── Agent loop ────────────────────────────────────────────────────────────
    agent_max_iterations: int = 10
    # Hard cap on ReAct loop iterations per request.
    # Prevents infinite loops if a tool keeps failing or Claude keeps calling tools.

    rag_top_k: int = 5
    # How many documents to retrieve from Milvus per RAG query.
    # More = more context for Claude, but also more tokens consumed.

    rag_min_score: float = 0.72
    # Minimum similarity score (0.0–1.0) to include a document in results.
    # 0.72 = only documents that are at least 72% semantically similar to the query.
    # Lower = more results but lower quality. Higher = fewer but more relevant results.

    # ── Optional: custom Anthropic endpoint (proxy / on-prem) ────────────────
    anthropic_base_url: str | None = None
    # Set ANTHROPIC_BASE_URL in .env to point Claude at a proxy or internal endpoint.
    # If not set, the official Anthropic API (https://api.anthropic.com) is used.

    # ── Optional integrations ─────────────────────────────────────────────────
    tavily_api_key: str | None = None
    # `str | None` = this field can be a string OR None (the Python equivalent of TS's `string | null`).
    # If TAVILY_API_KEY is not in .env, it defaults to None — web search is skipped gracefully.

    class Config:
        # Inner class that configures pydantic-settings behavior.
        env_file = ".env"
        # Also read from a .env file in the project root (for local development).
        # In production, real environment variables take priority over the .env file.


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiate Settings() ONCE when this module first imports.
# Any other file that does `from src.config.settings import settings`
# gets this same object — no re-reading of .env on every import.
settings = Settings()
# If any required field is missing (e.g. ANTHROPIC_API_KEY not set),
# pydantic raises a ValidationError HERE — before any request is processed.
