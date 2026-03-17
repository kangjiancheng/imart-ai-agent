from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    All configuration for the app-ai-base service.
    Reads values from environment variables and .env file via pydantic-settings.
    Fields without defaults are required; app fails fast with a clear error if missing.
    """

    # ── Claude ────────────────────────────────────────────────────────────────
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 4096

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Uses BAAI/bge-m3 (local, free, multilingual) — no API key required.
    # Model downloaded once via ModelScope into ~/.cache/modelscope on first run.

    # ── Milvus vector database ────────────────────────────────────────────────
    milvus_uri: str
    # Docker Milvus: http://localhost:19530
    # Milvus Lite (no Docker): ./milvus_local.db

    milvus_token: str = ""
    # Milvus server auth in "user:password" format (e.g. "root:Milvus").
    # Leave empty for Milvus Lite.

    milvus_collection_knowledge: str = "knowledge_base"
    milvus_collection_memory: str = "user_memory"

    # ── Agent loop ────────────────────────────────────────────────────────────
    agent_max_iterations: int = 10
    rag_top_k: int = 5
    rag_min_score: float = 0.50

    # ── Optional: custom Anthropic endpoint (proxy / on-prem) ────────────────
    anthropic_base_url: str | None = None

    # ── Optional integrations ─────────────────────────────────────────────────
    tavily_api_key: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
