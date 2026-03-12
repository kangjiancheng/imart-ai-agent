from pydantic import BaseModel, Field
from typing import Literal


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class UserContext(BaseModel):
    subscription_tier: str
    locale: str = "en-US"
    timezone: str = "UTC"


class AgentRequest(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1, max_length=10_000)
    history: list[HistoryMessage] = []
    user_context: UserContext
    session_id: str
    stream: bool = True
    document_context: str | None = None
