from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Claude
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 4096

    # Milvus vector database
    milvus_uri: str
    milvus_token: str = ""
    milvus_collection_knowledge: str = "knowledge_base"
    milvus_collection_memory: str = "user_memory"

    # Agent loop
    agent_max_iterations: int = 10
    rag_top_k: int = 5
    rag_min_score: float = 0.50

    # Optional
    anthropic_base_url: str | None = None
    tavily_api_key: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
