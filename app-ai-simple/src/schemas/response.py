from pydantic import BaseModel


class Document(BaseModel):
    content: str
    source: str
    score: float


class AgentResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []
