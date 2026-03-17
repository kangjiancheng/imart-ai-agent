from pydantic import BaseModel


class Document(BaseModel):
    """A single retrieved document chunk from the RAG knowledge base."""
    content: str
    source: str
    score: float


class AgentResponse(BaseModel):
    """Structured response for non-streaming agent results and OpenAPI docs."""
    answer: str
    session_id: str
    sources: list[str] = []
